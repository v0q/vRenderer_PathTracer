#pragma once

#include <cuda_runtime.h>
#include "vRenderer.h"

class vRendererCuda : public vRenderer
{
public:
  vRendererCuda();
  ~vRendererCuda();

  void init(const unsigned int &_w, const unsigned int &_h) override;
  void registerTextureBuffer(GLuint &_texture) override;
  void render() override;
  void cleanUp() override;
	void updateCamera(float *_cam, float *_dir);
  unsigned int getFrameCount() const override { return m_frame; }
private:
  void validateCuda(cudaError_t _err);

  cudaGraphicsResource_t m_cudaGLTextureBuffer;
  cudaArray *m_cudaImgArray;

  float3 *m_camera;
  float3 *m_camdir;
  float3 *m_colorArray;

  unsigned int m_width;
  unsigned int m_height;
  unsigned int m_frame;

  bool m_initialised;
};
