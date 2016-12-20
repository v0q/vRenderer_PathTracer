#pragma once

#include <cl/cl.hpp>
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
  unsigned int getFrameCount() const override { return m_frame; }
private:
  cl::Platform m_platform;
  cl::Device m_device;
  cl::Context m_context;
  cl::Program m_program;
  cl::Kernel m_kernel;
  cl::Memory m_glTexture;
  cl::Buffer m_colorArray;
  cl::CommandQueue m_queue;
  std::vector<cl::Memory> m_GLBuffers;

  cl_float3 m_camera;
  cl_float3 m_camdir;

  unsigned int m_width;
  unsigned int m_height;
  unsigned int m_frame;

  bool m_initialised;
};
