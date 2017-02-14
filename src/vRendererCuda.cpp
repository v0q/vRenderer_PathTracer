#include "vRendererCuda.h"

#include <chrono>
#include <cuda_gl_interop.h>

vRendererCuda::vRendererCuda() :
  m_frame(1),
	m_triCount(0),
	m_bvhBoxCount(0),
	m_triIdxCount(0),
	m_initialised(false)
{
  std::cout << "Cuda vRenderer ctor called\n";
}

vRendererCuda::~vRendererCuda()
{
  std::cout << "Cuda vRenderer dtor called\n";
  cleanUp();
}

void vRendererCuda::init(const unsigned int &_w, const unsigned int &_h)
{
  assert(_w != 0 && _h != 0);
  m_width = _w;
  m_height = _h;

  unsigned int sz = m_width*m_height;
	validateCuda(cudaMalloc(&m_colorArray, sizeof(float4)*sz), "Malloc pixel buffer");
	validateCuda(cudaMalloc(&m_camera, sizeof(float4)), "Malloc camera origin buffer");
	validateCuda(cudaMalloc(&m_camdir, sizeof(float4)), "Malloc camera direction buffer");

	float4 cam = make_float4(50.0f, 52.0f, 295.6f, 0.0f);
	float4 camdir = make_float4(0.0f, -0.0425734f, -0.999093f, 0.0f);

	validateCuda(cudaMemcpy(m_camera, &cam, sizeof(float4), cudaMemcpyHostToDevice), "Initialise camera origin buffer");
	validateCuda(cudaMemcpy(m_camdir, &camdir, sizeof(float4), cudaMemcpyHostToDevice), "Initialise camera direction buffer");
	cu_fillFloat4(m_colorArray, make_float4(0.0f, 0.0f, 0.0f, 0.0f), sz);
	cudaDeviceSynchronize();

//	cudaMemcpyToSymbol(kNumPlaneSetNormals, &BVH::m_numPlaneSetNormals, 1);

  m_initialised = true;
}

void vRendererCuda::registerTextureBuffer(GLuint &_texture)
{
  assert(m_initialised);

	validateCuda(cudaGraphicsGLRegisterImage(&m_cudaGLTextureBuffer, _texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
}

void vRendererCuda::updateCamera(const float *_cam, const float *_dir)
{
	if(_cam != nullptr)
	{
		validateCuda(cudaMemcpy(m_camera, _cam, sizeof(float4), cudaMemcpyHostToDevice), "Update camera origin buffer");
	}
	if(_dir != nullptr)
	{
		validateCuda(cudaMemcpy(m_camdir, _dir, sizeof(float4), cudaMemcpyHostToDevice), "Update camera direction buffer");
	}

	m_frame = 1;
	cu_fillFloat4(m_colorArray, make_float4(0.0f, 0.0f, 0.0f, 0.0f), m_width*m_height);
	cudaDeviceSynchronize();
}

unsigned int _triCount;
unsigned int _bvhBoxCount;
unsigned int _triIdxCount;

void vRendererCuda::render()
{
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	validateCuda(cudaGraphicsMapResources(1, &m_cudaGLTextureBuffer), "Map GL texture buffer");
	validateCuda(cudaGraphicsSubResourceGetMappedArray(&m_cudaImgArray, m_cudaGLTextureBuffer, 0, 0), "Attach the mapped texture buffer to a cuda resource");

  cudaResourceDesc wdsc;
  wdsc.resType = cudaResourceTypeArray;
  wdsc.res.array.array = m_cudaImgArray;
  cudaSurfaceObject_t writeSurface;
	validateCuda(cudaCreateSurfaceObject(&writeSurface, &wdsc), "Create a writeable cuda surface");
	cu_runRenderKernel(writeSurface,
										 m_triangleData,
										 m_triIdxList,
										 m_bvhLimits,
										 m_bvhChildrenOrTriangles,
										 m_triCount,
										 m_bvhBoxCount,
										 m_triIdxCount,
										 m_colorArray,
										 m_camera,
										 m_camdir,
										 m_width,
										 m_height,
										 m_frame++,
										 std::chrono::duration_cast<std::chrono::milliseconds>(t1.time_since_epoch()).count());
	validateCuda(cudaDestroySurfaceObject(writeSurface), "Clean up the surface");
	validateCuda(cudaGraphicsUnmapResources(1, &m_cudaGLTextureBuffer), "Free the mapped GL texture buffer");
	validateCuda(cudaStreamSynchronize(0), "Synchronize");
}

void vRendererCuda::cleanUp()
{
  if(m_initialised)
	{
		if(m_triangleData)
			validateCuda(cudaFree(m_triangleData), "Clean up triangle data buffer");
		if(m_triIdxList)
			validateCuda(cudaFree(m_triIdxList), "Clean up triangle index buffer");
		if(m_bvhLimits)
			validateCuda(cudaFree(m_bvhLimits), "Clean up BVH limits buffer");
		if(m_bvhChildrenOrTriangles)
			validateCuda(cudaFree(m_bvhChildrenOrTriangles), "Clean up BVH node buffer");

		validateCuda(cudaFree(m_colorArray), "Clean up color buffer");
		validateCuda(cudaFree(m_camera), "Clean up camera buffer");
		validateCuda(cudaFree(m_camdir), "Clean up camera direction buffer");
    cudaGraphicsUnregisterResource(m_cudaGLTextureBuffer);
  }
}

void vRendererCuda::initMesh(const vMeshData &_meshData)
{
	// Triangle data
	float4 *triData = new float4[_meshData.m_triangles.size() * 5];
	for(unsigned int i = 0; i < _meshData.m_triangles.size(); ++i)
	{
		triData[5 * i	+ 0].x = _meshData.m_triangles[i].m_center.x;
		triData[5 * i	+ 0].y = _meshData.m_triangles[i].m_center.y;
		triData[5 * i	+ 0].z = _meshData.m_triangles[i].m_center.z;
		triData[5 * i	+ 0].w = 0.f;

		triData[5 * i + 1].x = _meshData.m_triangles[i].m_normal.x;
		triData[5 * i + 1].y = _meshData.m_triangles[i].m_normal.y;
		triData[5 * i + 1].z = _meshData.m_triangles[i].m_normal.z;
		triData[5 * i + 1].w = _meshData.m_triangles[i].m_d;

		triData[5 * i + 2].x = _meshData.m_triangles[i].m_e1.x;
		triData[5 * i + 2].y = _meshData.m_triangles[i].m_e1.y;
		triData[5 * i + 2].z = _meshData.m_triangles[i].m_e1.z;
		triData[5 * i + 2].w = _meshData.m_triangles[i].m_d1;

		triData[5 * i + 3].x = _meshData.m_triangles[i].m_e2.x;
		triData[5 * i + 3].y = _meshData.m_triangles[i].m_e2.y;
		triData[5 * i + 3].z = _meshData.m_triangles[i].m_e2.z;
		triData[5 * i + 3].w = _meshData.m_triangles[i].m_d2;

		triData[5 * i + 4].x = _meshData.m_triangles[i].m_e3.x;
		triData[5 * i + 4].y = _meshData.m_triangles[i].m_e3.y;
		triData[5 * i + 4].z = _meshData.m_triangles[i].m_e3.z;
		triData[5 * i + 4].w = _meshData.m_triangles[i].m_d3;
	}

	validateCuda(cudaMalloc(&m_triangleData, _meshData.m_triangles.size() * 5 * sizeof(float4)), "Allocate memory for triangle data on GPU");
	validateCuda(cudaMemcpy(m_triangleData, triData, _meshData.m_triangles.size() * 5 * sizeof(float4), cudaMemcpyHostToDevice), "Copy triangle data to GPU");

	m_triCount = _meshData.m_triangles.size();

	delete [] triData;

	// Triangle indices
	validateCuda(cudaMalloc(&m_triIdxList, _meshData.m_cfbvhTriIndCount * sizeof(unsigned int)), "Allocate memory for triangle indices on GPU");
	validateCuda(cudaMemcpy(m_triIdxList, _meshData.m_cfbvhTriIndices, _meshData.m_cfbvhTriIndCount * sizeof(unsigned int), cudaMemcpyHostToDevice), "Copy triangle indices to GPU");

	m_triIdxCount = _meshData.m_cfbvhTriIndCount;

	// BVH Limits
	float2 *bvhLimits = new float2[_meshData.m_cfbvhBoxCount * 3];
	for(unsigned int i = 0; i < _meshData.m_cfbvhBoxCount; ++i)
	{
		bvhLimits[3 * i + 0].x = _meshData.m_cfbvh[i].m_bottom.x;
		bvhLimits[3 * i + 0].y = _meshData.m_cfbvh[i].m_top.x;

		bvhLimits[3 * i + 1].x = _meshData.m_cfbvh[i].m_bottom.y;
		bvhLimits[3 * i + 1].y = _meshData.m_cfbvh[i].m_top.z;

		bvhLimits[3 * i + 2].x = _meshData.m_cfbvh[i].m_bottom.y;
		bvhLimits[3 * i + 2].y = _meshData.m_cfbvh[i].m_top.z;
	}

	validateCuda(cudaMalloc(&m_bvhLimits, _meshData.m_cfbvhBoxCount * 3 * sizeof(float2)), "Allocate memory for bvh limits on GPU");
	validateCuda(cudaMemcpy(m_bvhLimits, bvhLimits, _meshData.m_cfbvhBoxCount * 3 * sizeof(float2), cudaMemcpyHostToDevice), "Copy bvh limits to GPU");

	m_bvhBoxCount = _meshData.m_cfbvhBoxCount;

	delete [] bvhLimits;

	// No need to have this and the limits in separate loops but makes it easier to follow
	uint4 *bvhChildrenOrTriangles = new uint4[_meshData.m_cfbvhBoxCount];
	for(unsigned int i = 0; i < _meshData.m_cfbvhBoxCount; ++i)
	{
		bvhChildrenOrTriangles[i].x = _meshData.m_cfbvh[i].m_u.m_leaf.m_count;
		bvhChildrenOrTriangles[i].y = _meshData.m_cfbvh[i].m_u.m_inner.m_rightIndex;
		bvhChildrenOrTriangles[i].z = _meshData.m_cfbvh[i].m_u.m_inner.m_leftIndex;
		bvhChildrenOrTriangles[i].w = _meshData.m_cfbvh[i].m_u.m_leaf.m_startIndexInTriIndexList;
	}

	validateCuda(cudaMalloc(&m_bvhChildrenOrTriangles, _meshData.m_cfbvhBoxCount * sizeof(uint4)), "Allocate memory for bvh child nodes and triangle data on GPU");
	validateCuda(cudaMemcpy(m_bvhChildrenOrTriangles, bvhChildrenOrTriangles, _meshData.m_cfbvhBoxCount * sizeof(uint4), cudaMemcpyHostToDevice), "Copy bvh child nodes and triangle data to GPU");

	delete [] bvhChildrenOrTriangles;
}

void vRendererCuda::validateCuda(cudaError_t _err, const std::string &_msg)
{
  if(_err != cudaSuccess)
  {
		std::cerr << "Failed to perform a cuda operation: " << _msg << "\n";
    exit(0);
  }
}
