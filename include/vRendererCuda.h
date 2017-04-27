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
	void registerDepthBuffer(GLuint &_depthTexture) override;
  void render() override;
  void cleanUp() override;
	void updateCamera() override;
	void initMesh(const vMeshData &_meshData) override;
	void initHDR(const Imf::Rgba *_colours, const unsigned int &_w, const unsigned int &_h);
	void loadTexture(const unsigned char *_texture, const unsigned int &_w, const unsigned int &_h, const unsigned int &_type) override;
	void clearBuffer() override;
  unsigned int getFrameCount() const override { return m_frame - 1; }
private:
	void validateCuda(cudaError_t _err, const std::string &_msg = "");

  cudaGraphicsResource_t m_cudaGLTextureBuffer;
	cudaGraphicsResource_t m_cudaGLDepthBuffer;
	cudaArray *m_cudaImgArray;
	cudaArray *m_cudaDepthArray;

	// Cuda buffers
	vCamera m_camera;
	float4 *m_colorArray;

	// Mesh buffers
	float4 *m_vertices;
	float4 *m_normals;
	float4 *m_bvhData;
	float2 *m_uvs;
	unsigned int *m_triIdxList;

	float4 *m_hdr;
	float4 *m_diffuse;
	float4 *m_normal;
	float4 *m_specular;

  unsigned int m_width;
  unsigned int m_height;
  unsigned int m_frame;
	unsigned int m_vertCount;
	unsigned int m_bvhNodeCount;
	unsigned int m_triIdxCount;

  bool m_initialised;
};
