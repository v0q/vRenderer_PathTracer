#pragma once

#include <cl/cl.hpp>
#include "vRenderer.h"

class vRendererCL : public vRenderer
{
public:
  vRendererCL();
  ~vRendererCL();

  void init() override;
  void registerTextureBuffer(GLint &_texture) override;
  void cleanUp() override;
private:
  cl::Platform m_platform;
  cl::Device m_device;
  cl::Context m_context;
  cl::Program m_program;
};
