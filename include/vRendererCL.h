#pragma once

#ifdef __APPLE__
	#include <cl/cl.hpp>
#else
	#include <cuda/CL/cl.hpp>
#endif

#include "vRenderer.h"

class vRendererCL : public vRenderer
{
private:
  struct vCamera {
    cl_float4 m_origin;
    cl_float4 m_dir;
    cl_float4 m_upV;
    cl_float4 m_rightV;
    float m_fovScale;
  };

public:
  vRendererCL();
  ~vRendererCL();

  void init(const unsigned int &_w, const unsigned int &_h) override;
  void registerTextureBuffer(GLuint &_texture) override;
  void render() override;
  void cleanUp() override;
  void updateCamera() override;
  void initMesh(const vMeshData &_meshData) override;
  void initHDR(const Imf::Rgba *_pixelBuffer, const unsigned int &_w, const unsigned int &_h) override;
  void clearBuffer() override;
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

  vCamera m_camera;

  // Mesh buffers
  cl::Buffer m_vertices;
  cl::Buffer m_normals;
  cl::Buffer m_bvhNodes;
  cl::Buffer m_triIdxList;

  // HDR map
  cl::Image2D m_hdr;

  unsigned int m_width;
  unsigned int m_height;
  unsigned int m_frame;

  bool m_initialised;
};
