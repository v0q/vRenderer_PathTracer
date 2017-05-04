///
/// \file vRendererCL.cpp
/// \brief Implements the OpenCL version of the renderer
///

/*
 * Ref: http://sa10.idav.ucdavis.edu/docs/sa10-dg-opencl-gl-interop.pdf
 */

#include <QImage>
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
	#undef CursorShape
#endif

#include "vRendererCL.h"

vRendererCL::vRendererCL() :
	m_brdf(nullptr),
  m_hdr(nullptr),
  m_diffuse(nullptr),
  m_normal(nullptr),
  m_specular(nullptr),
  m_frame(1),
  m_diffuseMapSet(false),
  m_normalMapSet(false),
  m_specularMapSet(false),
	m_useCornellBox(false),
	m_useExampleSphere(false),
	m_meshInitialised(false),
	m_viewBRDF(false),
	m_hasBRDF(false),
  m_initialised(false)
{
	m_fresnelCoef = 0.1f;
	m_fresnelPow = 3.f;
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

	///
	/// Get and create OpenCL/OpenGL interop context
	///

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
		std::cout << "\tMax image width: " << m_device.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>() << ", height: " << m_device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>() << "\n";
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
		std::cerr << buildlog << "\n";
		FILE *log = fopen("errorlog.txt", "w");
		fprintf(log, "%s\n", buildlog.c_str());

		std::cerr << "Build log saved to errorlog.txt\n";
		exit(EXIT_FAILURE);
	}

  m_kernel = cl::Kernel(m_program, "render");

	m_queue = cl::CommandQueue(m_context, m_device);
	m_colorArray = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, m_width*m_height*sizeof(cl_float3));

  cl_int err;
  cl::ImageFormat format;
  format.image_channel_data_type = CL_FLOAT;
  format.image_channel_order = CL_RGBA;

  float initImage[4] = {0.f, 0.f, 0.f, 0.f};

	// Need to allocate the buffers to something so the kernel can be enqueued
	m_diffuse = cl::Image2D(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, format, 1, 1, 0, initImage, &err);
	if(err != CL_SUCCESS)
	{
		std::cerr << "Failed to initialise diffuse map to GPU " << err << "\n";
		exit(EXIT_FAILURE);
	}

	m_normal = cl::Image2D(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, format, 1, 1, 0, initImage, &err);
	if(err != CL_SUCCESS)
	{
		std::cerr << "Failed to initialise diffuse map to GPU " << err << "\n";
		exit(EXIT_FAILURE);
	}

	m_specular = cl::Image2D(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, format, 1, 1, 0, initImage, &err);
	if(err != CL_SUCCESS)
	{
		std::cerr << "Failed to initialise diffuse map to GPU " << err << "\n";
		exit(EXIT_FAILURE);
	}

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

void vRendererCL::registerDepthBuffer(GLuint &_depthTexture)
{
	// Not implemented yet for OpenCL
}

void vRendererCL::cleanUp()
{
	// The OpenCL C++ wrapper handles the cleanup of OpenCL buffers automatically
  if(m_initialised)
  {
    m_GLBuffers.clear();
  }
}

void vRendererCL::updateCamera()
{
  m_virtualCamera->consume();

  m_camera.m_origin.x = m_virtualCamera->getOrig().m_x;
  m_camera.m_origin.y = m_virtualCamera->getOrig().m_y;
  m_camera.m_origin.z = m_virtualCamera->getOrig().m_z;
  m_camera.m_origin.w = 0.f;

  m_camera.m_dir.x = m_virtualCamera->getDir().m_x;
  m_camera.m_dir.y = m_virtualCamera->getDir().m_y;
  m_camera.m_dir.z = m_virtualCamera->getDir().m_z;
  m_camera.m_dir.w = 0.f;

  m_camera.m_upV.x = m_virtualCamera->getUp().m_x;
  m_camera.m_upV.y = m_virtualCamera->getUp().m_y;
  m_camera.m_upV.z = m_virtualCamera->getUp().m_z;
  m_camera.m_upV.w = 0.f;

  m_camera.m_rightV.x = m_virtualCamera->getRight().m_x;
  m_camera.m_rightV.y = m_virtualCamera->getRight().m_y;
  m_camera.m_rightV.z = m_virtualCamera->getRight().m_z;
  m_camera.m_rightV.w = 0.f;

  m_camera.m_fovScale = m_virtualCamera->getFovScale();

  m_frame = 1;
}

void vRendererCL::clearBuffer()
{
  m_frame = 1;
}

void vRendererCL::render()
{
  if(m_virtualCamera->isDirty())
  {
    updateCamera();
  }

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

	// Set kernel arguments
  m_kernel.setArg(0, m_glTexture);
  m_kernel.setArg(1, m_vertices);
  m_kernel.setArg(2, m_normals);
  m_kernel.setArg(3, m_tangents);
  m_kernel.setArg(4, m_bvhNodes);
  m_kernel.setArg(5, m_uvs);
	m_kernel.setArg(6, m_brdf);
	m_kernel.setArg(7, m_colorArray);
	m_kernel.setArg(8, m_hdr);
	m_kernel.setArg(9, m_diffuse);
	m_kernel.setArg(10, m_normal);
	m_kernel.setArg(11, m_specular);
	m_kernel.setArg(12, m_fresnelPow);
	m_kernel.setArg(13, m_fresnelCoef);
	m_kernel.setArg(14, (m_diffuseMapSet ? 1 : 0 ));
	m_kernel.setArg(15, (m_normalMapSet ? 1 : 0 ));
	m_kernel.setArg(16, (m_specularMapSet ? 1 : 0 ));
	m_kernel.setArg(17, (m_useCornellBox ? 1 : 0 ));
	m_kernel.setArg(18, (m_useExampleSphere ? 1 : 0 ));
	m_kernel.setArg(19, (m_meshInitialised ? 1 : 0 ));
	m_kernel.setArg(20, (m_viewBRDF ? 1 : 0 ));
	m_kernel.setArg(21, (m_hasBRDF ? 1 : 0 ));
	m_kernel.setArg(22, m_camera);
	m_kernel.setArg(23, m_width);
	m_kernel.setArg(24, m_height);
	m_kernel.setArg(25, m_frame++);
	m_kernel.setArg(26, static_cast<unsigned int>(std::chrono::duration_cast<std::chrono::milliseconds>(t1.time_since_epoch()).count()));

	// Run the trace step
  if((err = m_queue.enqueueNDRangeKernel(m_kernel, cl::NullRange, globalRange, localRange, nullptr, &event)) != CL_SUCCESS)
  {
    std::cout << "Failed to enqueue the kernel: " << err << "\n";
    exit(EXIT_FAILURE);
  }
	event.wait();

	cl::finish();

	// After we're finished, release the acquired GL objects
	if((err = m_queue.enqueueReleaseGLObjects(&m_GLBuffers, nullptr, &event)) != CL_SUCCESS)
	{
		std::cout << "Failed to release gl objects: " << err << "\n";
		exit(EXIT_FAILURE);
	}
	event.wait();
  t0 = t1;
}

void vRendererCL::initMesh(const vMeshData &_meshData)
{
	cl_int err;
  // Create a stack for node paired with an index
  std::vector<std::pair<const BVHNode *, unsigned int>> nodeStack{std::make_pair(_meshData.m_bvh.getRoot(), 0)};

  // Vector for bvh node data: child node/triangle indices, aabb's
  std::vector<cl_float4> bvhData;
  std::vector<cl_float4> verts;
  std::vector<cl_float4> normals;
  std::vector<cl_float4> tangents;
  std::vector<cl_float2> uvs;

  bvhData.resize(4);

  cl_float4 terminator;
  terminator.x = intAsFloat(0x80000000);
  terminator.y = 0.f;
  terminator.z = 0.f;
  terminator.w = 0.f;

  cl_float2 uvterminator;
  terminator.x = intAsFloat(0x80000000);
  terminator.y = 0.f;

	// Loop through the SBVH tree
  while(nodeStack.size())
  {
    const BVHNode *node = nodeStack.back().first;
    unsigned int idx = nodeStack.back().second;
    nodeStack.pop_back();

    AABB bounds[2];
    int indices[2];
//		if(!node->isLeaf())
//		{
		for(unsigned int i = 0; i < 2; ++i)
		{
			// Get the bounds of the node
			const BVHNode *child = node->childNode(i);
			bounds[i] = child->getBounds();

			if(!child->isLeaf())
			{
				// Index for the next node is an offset in memory
				unsigned int cidx = bvhData.size();
				indices[i] = cidx;// * sizeof(float4);
				nodeStack.push_back(std::make_pair(child, cidx));

				// Allocate space for the node data (e.g. bounds, indices etc)
				bvhData.resize(bvhData.size() + 4);
				continue;
			}

			const LeafNode *leaf = dynamic_cast<const LeafNode *>(child);
			// Triangle index stored as its complement to distinquish them from child nodes (e.g. ~0 = -1, ~1 = -2...)
			indices[i] = ~verts.size();
			for(unsigned int j = leaf->firstIndex(); j < leaf->lastIndex(); ++j)
			{
				unsigned int triInd = _meshData.m_bvh.getTriIndex(j);
				const vHTriangle tri = _meshData.m_triangles[triInd];
				// Get the triangle data and push it to their respective buffers
				for(unsigned int k = 0; k < 3; ++k)
				{
					const ngl::Vec3 &vert = _meshData.m_vertices[tri.m_indices[k]].m_vert;
					const ngl::Vec3 &norm = _meshData.m_vertices[tri.m_indices[k]].m_normal;
					const ngl::Vec3 &tangent = _meshData.m_vertices[tri.m_indices[k]].m_tangent;

					cl_float4 v;
					v.x = vert.m_x;
					v.y = vert.m_y;
					v.z = vert.m_z;
					v.w = 0.f;

					cl_float4 n;
					n.x = norm.m_x;
					n.y = norm.m_y;
					n.z = norm.m_z;
					n.w = 0.f;

					cl_float4 t;
					t.x = tangent.m_x;
					t.y = tangent.m_y;
					t.z = tangent.m_z;
					t.w = 0.f;

					cl_float2 uv;
					uv.x = _meshData.m_vertices[tri.m_indices[k]].m_u;
					uv.y = _meshData.m_vertices[tri.m_indices[k]].m_v;

					verts.push_back(v);
					normals.push_back(n);
					tangents.push_back(t);
					uvs.push_back(uv);
				}
			}
			// Terminate triangles
			verts.push_back(terminator);
			normals.push_back(terminator);
			tangents.push_back(terminator);
			uvs.push_back(uvterminator);
		}

		// Node bounding box
		// Stored int child 1 XY, child 2 XY, child 1 & 2 Z
		bvhData[idx + 0].x = bounds[0].minBounds().m_x;
		bvhData[idx + 0].y = bounds[0].maxBounds().m_x;
		bvhData[idx + 0].z = bounds[0].minBounds().m_y;
		bvhData[idx + 0].w = bounds[0].maxBounds().m_y;

		bvhData[idx + 1].x = bounds[1].minBounds().m_x;
		bvhData[idx + 1].y = bounds[1].maxBounds().m_x;
		bvhData[idx + 1].z = bounds[1].minBounds().m_y;
		bvhData[idx + 1].w = bounds[1].maxBounds().m_y;

		bvhData[idx + 2].x = bounds[0].minBounds().m_z;
		bvhData[idx + 2].y = bounds[0].maxBounds().m_z;
		bvhData[idx + 2].z = bounds[1].minBounds().m_z;
		bvhData[idx + 2].w = bounds[1].maxBounds().m_z;

		// Storing indices as floats
		bvhData[idx + 3].x = intAsFloat(indices[0]);
		bvhData[idx + 3].y = intAsFloat(indices[1]);
		bvhData[idx + 3].z = 0.f;
		bvhData[idx + 3].w = 0.f;
  }

  // Copy buffers to GPU
  m_vertices = cl::Buffer(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, verts.size()*sizeof(cl_float4), &verts[0], &err);

  if(err != CL_SUCCESS)
  {
    std::cerr << "Failed to copy vertex data to GPU " << err << "\n";
    exit(EXIT_FAILURE);
  }

  m_normals = cl::Buffer(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, normals.size()*sizeof(cl_float4), &normals[0], &err);

  if(err != CL_SUCCESS)
  {
    std::cerr << "Failed to copy normal data to GPU " << err << "\n";
    exit(EXIT_FAILURE);
  }

  m_tangents = cl::Buffer(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, tangents.size()*sizeof(cl_float4), &tangents[0], &err);

  if(err != CL_SUCCESS)
  {
    std::cerr << "Failed to copy normal data to GPU " << err << "\n";
    exit(EXIT_FAILURE);
  }

  m_uvs = cl::Buffer(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, uvs.size()*sizeof(cl_float4), &uvs[0], &err);

  if(err != CL_SUCCESS)
  {
    std::cerr << "Failed to copy normal data to GPU " << err << "\n";
    exit(EXIT_FAILURE);
  }

  m_bvhNodes = cl::Buffer(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, bvhData.size()*sizeof(cl_float4), &bvhData[0], &err);

  if(err != CL_SUCCESS)
  {
    std::cerr << "Failed to BVH data to GPU " << err << "\n";
    exit(EXIT_FAILURE);
  }

	m_meshInitialised = true;
}

void vRendererCL::loadHDR(const Imf::Rgba *_pixelBuffer, const unsigned int &_w, const unsigned int &_h)
{
  cl_int err;
  cl::ImageFormat format;
  format.image_channel_data_type = CL_HALF_FLOAT;
  format.image_channel_order = CL_RGBA;

	// Load the pixelbuffer data to the GPU
  m_hdr = cl::Image2D(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, format, _w, _h, 0, (void*)_pixelBuffer, &err);

  if(err != CL_SUCCESS)
  {
    std::cerr << "Failed to load HDR map to GPU " << err << "\n";
    exit(EXIT_FAILURE);
  }
}


void vRendererCL::loadTexture(const QImage &_texture, const float &_gamma, const unsigned int &_type)
{
	unsigned int w = _texture.width();
	unsigned int h = _texture.height();
	float correction = (_gamma > 0.001f ? 1.f/_gamma : 1.f);

	float *dataAsFloats = new float[w*h*4];

  cl_int err = CL_SUCCESS;
  cl::ImageFormat format;
  format.image_channel_data_type = CL_FLOAT;
  format.image_channel_order = CL_RGBA;

	for(unsigned int j = 0; j < h; ++j)
	{
		for(unsigned int i = 0; i < w; ++i)
		{
			QColor pixel(_texture.pixel(i, j));
			if(_type == DIFFUSE)
			{
				// Invert gamma correction if it's already applied
				dataAsFloats[i*4 + j*w*4 + 0] = std::pow(pixel.red()/255.f, correction);
				dataAsFloats[i*4 + j*w*4 + 1] = std::pow(pixel.green()/255.f, correction);
				dataAsFloats[i*4 + j*w*4 + 2] = std::pow(pixel.blue()/255.f, correction);
				dataAsFloats[i*4 + j*w*4 + 3] = pixel.alpha()/255.f;
			}
			else
			{
				dataAsFloats[i*4 + j*w*4 + 0] = pixel.red()/255.f;
				dataAsFloats[i*4 + j*w*4 + 1] = pixel.green()/255.f;
				dataAsFloats[i*4 + j*w*4 + 2] = pixel.blue()/255.f;
				dataAsFloats[i*4 + j*w*4 + 3] = pixel.alpha()/255.f;
			}
		}
	}

	// Load the data to the correct buffer
  switch(_type)
  {
    case DIFFUSE:
		{
			m_diffuse = cl::Image2D(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, format, w, h, 0, (void*)dataAsFloats, &err);
      m_diffuseMapSet = true;
    } break;
    case NORMAL:
		{
			m_normal = cl::Image2D(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, format, w, h, 0, (void*)dataAsFloats, &err);
      m_normalMapSet = true;
    } break;
    case SPECULAR:
		{
			m_specular = cl::Image2D(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, format, w, h, 0, (void*)dataAsFloats, &err);
      m_specularMapSet = true;
    } break;
    default: break;
  }

  if(err != CL_SUCCESS)
  {
		std::cerr << "Failed to load a texture map to GPU " << err << "\n";
    exit(EXIT_FAILURE);
  }

	// Delete the buffer from the CPU as it's no longer needed
  delete [] dataAsFloats;
}

bool vRendererCL::loadBRDF(const float *_brdf)
{
	// If the brdf data is valid (e.g. not a nullptr), load it to the GPU
	if(_brdf)
	{
		unsigned int n = BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D * BRDF_SAMPLING_RES_PHI_D / 2;

		cl_int err;
		// Copy brdf to GPU
		m_brdf = cl::Buffer(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, n * 3 * sizeof(float), const_cast<float *>(_brdf), &err);

		if(err != CL_SUCCESS)
		{
			std::cerr << "Failed to copy vertex data to GPU " << err << "\n";
			exit(EXIT_FAILURE);
		}

		m_hasBRDF = true;

		// Free the CPU memory
		delete [] _brdf;
		return true;
	}
	else
	{
		return false;
	}
}

void vRendererCL::useBRDF(const bool &_newVal)
{
	m_viewBRDF = _newVal;
}

void vRendererCL::useExampleSphere(const bool &_newVal)
{
	m_useExampleSphere = _newVal;
}

void vRendererCL::useCornellBox(const bool &_newVal)
{
	m_useCornellBox = _newVal;
}

float vRendererCL::intAsFloat(const unsigned int &_v)
{
  union
  {
    unsigned int a;
    float b;
  } a;
  a.a = _v;

  return a.b;
}
