#pragma once

#ifdef __APPLE__
	#include <cl/cl.hpp>
#else
	#include <cuda/CL/cl.hpp>
#endif

#include "vRenderer.h"

typedef struct vBoundingBox {
  cl_float2 m_x;
  cl_float2 m_y;
  cl_float2 m_z;
} vBoundingBox;

typedef struct vVert {
	cl_float4 m_vert;
	cl_float4 m_normal;
} vVert;

typedef struct vTriangle {
  vVert m_v1;
  vVert m_v2;
  vVert m_v3;
} vTriangle;

typedef struct vMesh {
  vTriangle *m_mesh;
  vBoundingBox m_bb;
  unsigned int m_triCount;
} vMesh;

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
  cl::Image1D m_triangleData;
  cl::Image1D m_bvhLimits;
  cl::Image1D m_bvhChildrenOrTriangles;
  cl::Image1D m_triIdxList;

  unsigned int m_width;
  unsigned int m_height;
  unsigned int m_frame;
  unsigned int m_triCount;
  unsigned int m_bvhBoxCount;
  unsigned int m_triIdxCount;

  bool m_initialised;
};
