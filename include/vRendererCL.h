#pragma once

#include <cl/cl.hpp>
#include "vRenderer.h"

struct float3
{
  float x;
  float y;
  float z;
};

class vRendererCL : public vRenderer
{
public:
  vRendererCL();
  ~vRendererCL();

  void init(const unsigned int &_w, const unsigned int &_h) override;
  void registerTextureBuffer(GLuint &_texture) override;
  void render() override;
  void cleanUp() override;
private:
  cl::Platform m_platform;
  cl::Device m_device;
  cl::Context m_context;
  cl::Program m_program;

  float3 *m_camera;
  float3 *m_camdir;
  float3 *m_colorArray;

  unsigned int m_width;
  unsigned int m_height;

  bool m_initialised;
};
