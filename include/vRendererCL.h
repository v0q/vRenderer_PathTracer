#pragma once

#ifdef __APPLE__
	#include <cl/cl.hpp>
#else
	#include <cuda/CL/cl.hpp>
#endif

#include "vRenderer.h"

class vRendererCL : public vRenderer
{
public:
  vRendererCL();
  ~vRendererCL();

  void init(const unsigned int &_w, const unsigned int &_h) override;
  void registerTextureBuffer(GLuint &_texture) override;
  void render() override;
  void cleanUp() override;
  void updateCamera(const float *_cam = nullptr, const float *_dir = nullptr) override;
  void initMesh(const vMeshData &_meshData) override;
  unsigned int getFrameCount() const override { return m_frame - 1; }
private:
  float intAsFloat(const int &_v);
  cl::Platform m_platform;
  cl::Device m_device;
  cl::Context m_context;
  cl::Program m_program;
  cl::Kernel m_kernel;
  cl::Memory m_glTexture;
  cl::Buffer m_colorArray;
  cl::CommandQueue m_queue;
	std::vector<cl::Memory> m_GLBuffers;

	cl_float4 m_camera;
	cl_float4 m_camdir;

  // Mesh buffers
  cl::Image1D m_vertices;
  cl::Image1D m_normals;
  cl::Image1D m_bvhNodes;
  cl::Image1D m_triIdxList;

  unsigned int m_width;
  unsigned int m_height;
  unsigned int m_frame;
  unsigned int m_triCount;
  unsigned int m_bvhBoxCount;
  unsigned int m_triIdxCount;

  bool m_initialised;
};
