#pragma once

#include <cuda_runtime.h>
#include "vRenderer.h"
#include "PathTracer.cuh"

class vRendererCuda : public vRenderer
{
public:
  vRendererCuda();
  ~vRendererCuda();

  void init(const unsigned int &_w, const unsigned int &_h) override;
  void registerTextureBuffer(GLuint &_texture) override;
  void render() override;
  void cleanUp() override;
  void updateCamera(const float *_cam = nullptr, const float *_dir = nullptr) override;
	void initMesh(const std::vector<vFloat3> &_vertData) override;
  unsigned int getFrameCount() const override { return m_frame - 1; }
private:
	void validateCuda(cudaError_t _err, const std::string &_msg = "");

  cudaGraphicsResource_t m_cudaGLTextureBuffer;
  cudaArray *m_cudaImgArray;
	std::vector<vMesh *> m_meshes;

	float4 *m_camera;
	float4 *m_camdir;
	float4 *m_colorArray;

  unsigned int m_width;
  unsigned int m_height;
  unsigned int m_frame;
	unsigned int m_triCount;

  bool m_initialised;
};
