/*
 * Ref: http://sa10.idav.ucdavis.edu/docs/sa10-dg-opencl-gl-interop.pdf
 */

#include <iostream>
#include <string>
#include <fstream>
#include <streambuf>
#include <vector>

#ifdef __APPLE__
  #include <GL/glew.h>
  #include <OpenCL/cl_gl_ext.h>
  #include <OpenGL.h>
#endif

#include "vRendererCL.h"

vRendererCL::vRendererCL() :
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

  std::ifstream clFile("cl/src/CL_UVRender.cl");
  if(!clFile)
  {
    std::cerr << "Could not find 'cl/src/CL_UVRender.cl'\n";
    exit(0);
  }
  std::string pathTracerSrc((std::istreambuf_iterator<char>(clFile)),
                             std::istreambuf_iterator<char>());

  m_program = cl::Program(m_context, pathTracerSrc.c_str());
  cl_int result = m_program.build({ m_device });
  if(result)
  {
    std::cerr << "Failed compile the program: " << result << "\n";
    if(result == CL_BUILD_PROGRAM_FAILURE)
    {
      std::string buildlog = m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device);
      FILE *log = fopen("errorlog.txt", "w");
      fprintf(log, "%s\n", buildlog.c_str());

      std::cerr << "Build log saved to errorlog.txt:\n";
    }
    exit(EXIT_FAILURE);
  }

  m_kernel = cl::Kernel(m_program, "render");
  m_queue = cl::CommandQueue(m_context, m_device);

  m_initialised = true;
}

void vRendererCL::registerTextureBuffer(GLuint &_texture)
{
  assert(m_initialised);
  cl_int err;
  m_colorArray = cl::ImageGL(m_context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, _texture, &err);
  if(err != CL_SUCCESS)
  {
    std::cout << "Failed to create OpenGL texture reference!" << err << "\n";
    exit(EXIT_FAILURE);
  }
  m_GLBuffers.push_back(m_colorArray);
}

void vRendererCL::render()
{
  cl::Event event;
  cl::NDRange globalRange = cl::NDRange(256, 256);
  cl::NDRange localRange = cl::NDRange(16, 16);
  cl_int err;
  std::cout << "Rendering...\n";

  if((err = m_queue.enqueueAcquireGLObjects(&m_GLBuffers, nullptr, &event)) != CL_SUCCESS)
  {
    std::cout << "Failed to acquire gl objects: " << err << "\n";
    exit(EXIT_FAILURE);
  }

  event.wait();

  m_kernel.setArg(0, &m_colorArray);
  m_kernel.setArg(0, m_width);
  m_kernel.setArg(1, m_height);
  if((err = m_queue.enqueueNDRangeKernel(m_kernel, cl::NullRange, globalRange, localRange, nullptr, &event)) != CL_SUCCESS)
  {
    std::cout << "Failed to enqueue the kernel: " << err << "\n";
    exit(EXIT_FAILURE);
  }
  event.wait();

  if((err = m_queue.enqueueReleaseGLObjects(&m_GLBuffers, nullptr, &event)) != CL_SUCCESS)
  {
    std::cout << "Failed to release gl objects: " << err << "\n";
    exit(EXIT_FAILURE);
  }
  event.wait();
}

void vRendererCL::cleanUp()
{
  if(m_initialised)
  {
    m_GLBuffers.clear();
  }
}
