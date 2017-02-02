/*
 * Ref: http://sa10.idav.ucdavis.edu/docs/sa10-dg-opencl-gl-interop.pdf
 */

#include <iostream>
#include <string>
#include <fstream>
#include <streambuf>
#include <vector>
#include <chrono>


#ifdef __APPLE__
  #include <GL/glew.h>
  #include <OpenCL/cl_gl_ext.h>
  #include <OpenGL.h>
#else
	#include <cuda/CL/cl_gl_ext.h>
	#include <GL/glew.h>
	#include <GL/glx.h>
#endif

#include "vRendererCL.h"

vRendererCL::vRendererCL() :
  m_frame(1),
  m_triCount(0),
  m_initialised(false)
{
  std::cout << "OpenCL vRenderer ctor called\n";
  m_GLBuffers.clear();
}

vRendererCL::~vRendererCL()
{
  std::cout << "OpenCL vRenderer dtor called\n";
  cleanUp();
}

void vRendererCL::init(const unsigned int &_w, const unsigned int &_h)
{
  assert(_w != 0 && _h != 0);
  m_width = _w;
	m_height = _h;

	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	std::cout << "Available platforms: \n";
	for(auto &p : platforms)
		std::cout << "\t" << p.getInfo<CL_PLATFORM_NAME>() << "\n";

	if(platforms.size())
	{
		std::cout << "Using platform: \n";
		m_platform = platforms[platforms.size() - 1];
		std::cout << "\t" << m_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
	}

	// Getting GPU devices
	std::vector<cl::Device> devices;
	m_platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

	std::cout << "Available devices: \n";
	for(auto &d : devices)
		std::cout << "\t" << d.getInfo<CL_DEVICE_NAME>() << "\n";

	if(devices.size())
	{
		std::cout << "Using device: \n";
		m_device = devices[devices.size() - 1];
		std::cout << "\t" << m_device.getInfo<CL_DEVICE_NAME>() << "\n";
	}

	std::cout << m_device.getInfo<CL_DEVICE_EXTENSIONS>() << "\n";

#ifdef __APPLE__
	CGLContextObj kCGLContext = CGLGetCurrentContext();
	CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
	cl_context_properties properties[3] =
	{
		CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
		(cl_context_properties)kCGLShareGroup,
		0
	};
#else
	cl_context_properties properties[7] =
	{
		CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
		CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
		CL_CONTEXT_PLATFORM, (cl_context_properties)(m_platform)(),
		0
	};
#endif

	m_context = cl::Context(m_device, properties);

	std::ifstream clFile("cl/src/PathTracer.cl");
	if(!clFile)
	{
		std::cerr << "Could not find 'cl/src/PathTracer.cl'\n";
		exit(0);
	}
	std::string pathTracerSrc((std::istreambuf_iterator<char>(clFile)),
														 std::istreambuf_iterator<char>());

	m_program = cl::Program(m_context, pathTracerSrc.c_str());
	cl_int result = m_program.build({ m_device }, "-cl-fast-relaxed-math");
	if(result)
	{
		std::cerr << "Failed compile the program: " << result << "\n";
		std::string buildlog = m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device);
		FILE *log = fopen("errorlog.txt", "w");
		fprintf(log, "%s\n", buildlog.c_str());

		std::cerr << "Build log saved to errorlog.txt:\n";
		exit(EXIT_FAILURE);
	}

	m_kernel = cl::Kernel(m_program, "render");
	m_queue = cl::CommandQueue(m_context, m_device);
	m_colorArray = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, m_width*m_height*sizeof(cl_float3));

  m_camera.x = 50.f;
  m_camera.y = 52.f;
  m_camera.z = 295.6f;
	m_camera.w = 0.f;

  m_camdir.x = 0.f;
	m_camdir.y = -0.0425734f;
	m_camdir.z = -0.999093f;
	m_camdir.w = 0.f;

  m_initialised = true;
}

void vRendererCL::registerTextureBuffer(GLuint &_texture)
{
	assert(m_initialised);
	cl_int err;
	m_glTexture = cl::ImageGL(m_context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, _texture, &err);
	if(err != CL_SUCCESS)
	{
		std::cout << "Failed to create OpenGL texture reference!" << err << "\n";
		exit(EXIT_FAILURE);
	}
	m_GLBuffers.push_back(m_glTexture);
}

void vRendererCL::render()
{
	cl::Event event;
	cl::NDRange globalRange = cl::NDRange(m_width, m_height);
	cl::NDRange localRange = cl::NDRange(16, 16);
	cl_int err;

	glFinish();
	static std::chrono::high_resolution_clock::time_point t0;
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
//  std::cout << "Rendering...\n";

	if((err = m_queue.enqueueAcquireGLObjects(&m_GLBuffers, nullptr, &event)) != CL_SUCCESS)
	{
		std::cout << "Failed to acquire gl objects: " << err << "\n";
		exit(EXIT_FAILURE);
	}

	event.wait();

	m_kernel.setArg(0, m_glTexture);
	if(m_meshes.size())
		m_kernel.setArg(1, m_meshes[0]);
	m_kernel.setArg(2, m_triCount);
	m_kernel.setArg(3, m_colorArray);
	m_kernel.setArg(4, m_camera);
	m_kernel.setArg(5, m_camdir);
	m_kernel.setArg(6, m_width);
	m_kernel.setArg(7, m_height);
	m_kernel.setArg(8, m_frame++);
	m_kernel.setArg(9, static_cast<unsigned int>(std::chrono::duration_cast<std::chrono::milliseconds>(t1.time_since_epoch()).count()));

	if((err = m_queue.enqueueNDRangeKernel(m_kernel, cl::NullRange, globalRange, localRange, nullptr, &event)) != CL_SUCCESS)
	{
		std::cout << "Failed to enqueue the kernel: " << err << "\n";
		exit(EXIT_FAILURE);
	}
	event.wait();

	cl::finish();

	if((err = m_queue.enqueueReleaseGLObjects(&m_GLBuffers, nullptr, &event)) != CL_SUCCESS)
	{
		std::cout << "Failed to release gl objects: " << err << "\n";
		exit(EXIT_FAILURE);
	}
	event.wait();
	t0 = t1;
}

void vRendererCL::cleanUp()
{
  if(m_initialised)
  {
    m_GLBuffers.clear();
  }
}

void vRendererCL::updateCamera(const float *_cam, const float *_dir)
{
  if(_cam != nullptr)
  {
    m_camera.x = _cam[0];
    m_camera.y = _cam[1];
    m_camera.z = _cam[2];
		m_camera.w = 0.f;
  }
  if(_dir != nullptr)
  {
    m_camdir.x = _dir[0];
    m_camdir.y = _dir[1];
    m_camdir.z = _dir[2];
		m_camdir.w = 0.f;
  }

  m_frame = 1;
}

void vRendererCL::initMesh(const std::vector<vFloat3> &_vertData)
{
	cl_int err;
	unsigned int sz = _vertData.size()/6;

	float scale = 15.f;
	float offset = 50.f;

	m_triCount = sz;
	vTriangle *triangles = new vTriangle[sz];

	for(unsigned int i = 0; i < _vertData.size(); i += 6)
	{
		vVert v1, v2, v3;
		v1.m_vert.x = _vertData[i].x * scale + offset;
		v1.m_vert.y = _vertData[i].y * scale + offset/2;
		v1.m_vert.z = _vertData[i].z * scale + offset;
		v1.m_vert.w = 0.f;
		v1.m_normal.x = _vertData[i+1].x;
		v1.m_normal.y = _vertData[i+1].y;
		v1.m_normal.z = _vertData[i+1].z;
		v1.m_normal.w = 0.f;
		v2.m_vert.x = _vertData[i+2].x * scale + offset;
		v2.m_vert.y = _vertData[i+2].y * scale + offset/2;
		v2.m_vert.z = _vertData[i+2].z * scale + offset;
		v2.m_vert.w = 0.f;
		v2.m_normal.x = _vertData[i+3].x;
		v2.m_normal.y = _vertData[i+3].y;
		v2.m_normal.z = _vertData[i+3].z;
		v2.m_normal.w = 0.f;
		v3.m_vert.x = _vertData[i+4].x * scale + offset;
		v3.m_vert.y = _vertData[i+4].y * scale + offset/2;
		v3.m_vert.z = _vertData[i+4].z * scale + offset;
		v3.m_vert.w = 0.f;
		v3.m_normal.x = _vertData[i+5].x;
		v3.m_normal.y = _vertData[i+5].y;
		v3.m_normal.z = _vertData[i+5].z;
		v3.m_normal.w = 0.f;
		triangles[i/6].m_v1 = v1;
		triangles[i/6].m_v2 = v2;
		triangles[i/6].m_v3 = v3;
	}
	m_meshes.push_back(cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz*sizeof(vTriangle), &triangles[0], &err));

	delete [] triangles;
}
