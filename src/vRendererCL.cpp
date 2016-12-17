#include <iostream>
#include <string>
#include <fstream>
#include <streambuf>
#include <vector>

#include "vRendererCL.h"

vRendererCL::vRendererCL() :
  m_initialised(false)
{
  std::cout << "OpenCL vRenderer ctor called\n";
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

  m_context = cl::Context(m_device);

//  std::ifstream clFile("cl/src/PathTracer.cl");
//  if(!clFile)
//  {
//    std::cerr << "Could not find 'cl/src/PathTracer.cl'\n";
//    exit(0);
//  }
//  std::string pathTracerSrc((std::istreambuf_iterator<char>(clFile)),
//                             std::istreambuf_iterator<char>());

//  m_program = cl::Program(m_context, pathTracerSrc.c_str());
//  cl_int result = m_program.build({ m_device });
//  if(result)
//  {
//    std::cerr << "Failed compile the program: " << result << "\n";
//    if(result == CL_BUILD_PROGRAM_FAILURE)
//    {
//      std::string buildlog = m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device);
//      FILE *log = fopen("errorlog.txt", "w");
//      fprintf(log, "%s\n", buildlog.c_str());

//      std::cerr << "Build log saved to errorlog.txt:\n";
//    }
//    exit(0);
//  }

  m_initialised = true;
}

void vRendererCL::registerTextureBuffer(GLuint &_texture)
{
  assert(m_initialised);

  (void)_texture;
}

void vRendererCL::render()
{
  std::cout << "Rendering...\n";
}

void vRendererCL::cleanUp()
{
  if(m_initialised)
  {

  }
}
