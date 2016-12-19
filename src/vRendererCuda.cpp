#include <chrono>
#include <cuda_gl_interop.h>

#include "vRendererCuda.h"
#include "PathTracer.cuh"

vRendererCuda::vRendererCuda() :
  m_frame(0),
  m_initialised(false)
{
  std::cout << "Cuda vRenderer ctor called\n";
}

vRendererCuda::~vRendererCuda()
{
  std::cout << "Cuda vRenderer dtor called\n";
  cleanUp();
}

void vRendererCuda::init(unsigned int &_w, unsigned int &_h)
{
  assert(_w != 0 && _h != 0);
  m_width = _w;
  m_height = _h;

  unsigned int sz = m_width*m_height;
  validateCuda(cudaMalloc(&m_colorArray, sizeof(float3)*sz));
  validateCuda(cudaMalloc(&m_camera, sizeof(float3)));
  validateCuda(cudaMalloc(&m_camdir, sizeof(float3)));

  float3 cam = make_float3(50, 52, 295.6);
  float3 camdir = make_float3(0, -0.042612, -1);

  validateCuda(cudaMemcpy(m_camera, &cam, sizeof(float3), cudaMemcpyHostToDevice));
  validateCuda(cudaMemcpy(m_camdir, &camdir, sizeof(float3), cudaMemcpyHostToDevice));
  cu_fillFloat3(m_colorArray, make_float3(0.0f, 0.0f, 0.0f), sz);

  m_initialised = true;
}

void vRendererCuda::registerTextureBuffer(GLuint &_texture)
{
  assert(m_initialised);

  validateCuda(cudaGraphicsGLRegisterImage(&m_cudaGLTextureBuffer, _texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
}

void vRendererCuda::render()
{
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  std::cout << "Rendering...\n";

  validateCuda(cudaGraphicsMapResources(1, &m_cudaGLTextureBuffer));
  validateCuda(cudaGraphicsSubResourceGetMappedArray(&m_cudaImgArray, m_cudaGLTextureBuffer, 0, 0));

  cudaResourceDesc wdsc;
  wdsc.resType = cudaResourceTypeArray;
  wdsc.res.array.array = m_cudaImgArray;
  cudaSurfaceObject_t writeSurface;
  validateCuda(cudaCreateSurfaceObject(&writeSurface, &wdsc));
  cu_ModifyTexture(writeSurface,
                   m_colorArray,
                   m_camera,
                   m_camdir,
                   width(), height(), m_frame++, std::chrono::duration_cast<std::chrono::milliseconds>(t1.time_since_epoch()).count());
  validateCuda(cudaDestroySurfaceObject(writeSurface));
  validateCuda(cudaGraphicsUnmapResources(1, &m_cudaGLTextureBuffer));
  validateCuda(cudaStreamSynchronize(0));
}

void vRendererCuda::cleanUp()
{
  if(m_initialised)
  {
    cudaFree(m_colorArray);
    cudaFree(m_camera);
    cudaFree(m_camdir);
    cudaGraphicsUnregisterResource(m_cudaGLTextureBuffer);
  }
}

void vRendererCuda::validateCuda(int _err)
{
  if(_err != cudaSuccess)
  {
    std::cerr << "Failed to perform a cuda operation\n";
    exit(0);
  }
}
