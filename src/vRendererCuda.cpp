#include "vRendererCuda.h"

#include <chrono>
#include <cuda_gl_interop.h>

#include "Utilities.cuh"

vRendererCuda::vRendererCuda() :
  m_frame(1),
	m_vertCount(0),
	m_bvhNodeCount(0),
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
										 m_vertices,
										 m_bvhData,
										 m_triIdxList,
										 m_vertCount,
										 m_bvhNodeCount,
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
		if(m_vertices)
			validateCuda(cudaFree(m_vertices), "Clean up triangle data buffer");
		if(m_triIdxList)
			validateCuda(cudaFree(m_triIdxList), "Clean up triangle index buffer");
		if(m_bvhData)
			validateCuda(cudaFree(m_bvhData), "Clean up BVH node buffer");

		validateCuda(cudaFree(m_colorArray), "Clean up color buffer");
		validateCuda(cudaFree(m_camera), "Clean up camera buffer");
		validateCuda(cudaFree(m_camdir), "Clean up camera direction buffer");
    cudaGraphicsUnregisterResource(m_cudaGLTextureBuffer);
  }
}

void vRendererCuda::initMesh(const vMeshData &_sbvhData)
{
//	std::vector<std::pair<const SBVHNode *, unsigned int>> stack{std::make_pair(_sbvhData.m_sbvh.getRoot(), 0)};
//	std::vector<float4> bvhData;
//	std::vector<float4> verts;
//	std::vector<unsigned int> triIndices;

//	bvhData.resize(4);

//	while(stack.size())
//	{
//		const SBVHNode *node = stack.back().first;
//		unsigned int idx = stack.back().second;
//		stack.pop_back();

//		AABB bounds[2];
//		int indices[2];
//		if(!node->isLeaf())
//		{
//			for(unsigned int i = 0; i < 2; ++i)
//			{
//				const SBVHNode *child = node->childNode(i);
//				bounds[i] = child->getBounds();

//				if(!child->isLeaf())
//				{
//					unsigned int cidx = bvhData.size() * sizeof(float4);
//					indices[i] = cidx;
//					stack.push_back(std::make_pair(child, cidx));

//					bvhData.resize(bvhData.size() + 4);
//					continue;
//				}

//				const LeafNode *leaf = dynamic_cast<const LeafNode *>(child);
//				indices[i] = ~triIndices.size();
//				for(unsigned int j = leaf->firstIndex(); j < leaf->lastIndex(); ++j)
//				{
//					unsigned int triInd = _sbvhData.m_sbvh.getTriIndex(j);
//					for(unsigned int k = 0; k < 3; ++k)
//					{
//						const ngl::Vec3 &vert = _sbvhData.m_vertices[_sbvhData.m_triangles[triInd].m_indices[k]];
//						verts.push_back(make_float4(vert.m_x, vert.m_y, vert.m_z, 0.f));
//					}
////					std::cout << "Triangle index: " << _sbvhData.m_sbvh.getTriIndex(j) << "\n";
//					triIndices.push_back(triInd);
//				}
//				// Terminate triangle
//				verts.push_back(make_float4(intAsFloat(0x80000000), 0, 0, 0));
//			}
//		}
//		else
//		{
//			bounds[0] = node->getBounds();
//			const LeafNode *leaf = dynamic_cast<const LeafNode *>(node);
//			indices[0] = ~triIndices.size();
//			indices[1] = 0x80000000;
//			for(unsigned int j = leaf->firstIndex(); j < leaf->lastIndex(); ++j)
//			{
//				unsigned int triInd = _sbvhData.m_sbvh.getTriIndex(j);
//				for(unsigned int k = 0; k < 3; ++k)
//				{
//					const ngl::Vec3 &vert = _sbvhData.m_vertices[_sbvhData.m_triangles[triInd].m_indices[k]];
//					verts.push_back(make_float4(vert.m_x, vert.m_y, vert.m_z, 0.f));
//				}
//				triIndices.push_back(triInd);
//			}
//			// Terminate triangle
//			verts.push_back(make_float4(intAsFloat(0x80000000), 0, 0, 0));
//		}
//		// Node bounding box
//		// Stored int child 1 XY, child 2 XY, child 1 & 2 Z
//		bvhData[idx*4 + 0] = make_float4(bounds[0].minBounds().m_x, bounds[0].maxBounds().m_x, bounds[0].minBounds().m_y, bounds[0].maxBounds().m_y);
//		bvhData[idx*4 + 1] = make_float4(bounds[1].minBounds().m_x, bounds[1].maxBounds().m_x, bounds[1].minBounds().m_y, bounds[1].maxBounds().m_y);
//		bvhData[idx*4 + 2] = make_float4(bounds[0].minBounds().m_z, bounds[0].maxBounds().m_z, bounds[1].minBounds().m_z, bounds[1].maxBounds().m_z);

//		// Doing "trickery" and storing our indices as floats by not mangling with the bits:
//		// 1 (int) = 0x00000001
//		// 1.0 (float) = 0x3F800000
//		// 0x00000001 = 1.40129846432481707092372958329E-45 (float)
////		std::cout << intAsFloat(indices[0] << " " << indices[1] << "\n";
//		bvhData[idx*4 + 3] = make_float4(intAsFloat(indices[0]), intAsFloat(indices[1]), 0, 0);
//	}

//	validateCuda(cudaMalloc(&m_vertices, verts.size()*sizeof(float4)), "Malloc vertex device pointer");
//	validateCuda(cudaMemcpy(m_vertices, &verts[0], verts.size()*sizeof(float4), cudaMemcpyHostToDevice), "Copy vertex data to gpu");

//	std::cout << verts.size() << "\n";

//	validateCuda(cudaMalloc(&m_bvhData, bvhData.size()*sizeof(float4)), "Malloc BVH node device pointer");
//	validateCuda(cudaMemcpy(m_bvhData, &bvhData[0], bvhData.size()*sizeof(float4), cudaMemcpyHostToDevice), "Copy bvh node data to gpu");

//	validateCuda(cudaMalloc(&m_triIdxList, triIndices.size()*sizeof(unsigned int)), "Malloc triangle index device pointer");
//	validateCuda(cudaMemcpy(m_triIdxList, &triIndices[0], triIndices.size()*sizeof(unsigned int), cudaMemcpyHostToDevice), "Copy triangle indices to gpu");

////	for(unsigned int i = 0; i < bvhData.size(); i += 4)
////	{
////		std::cout << "BVHData " << bvhData[i*4 + 3].x << " " << bvhData[i*4 + 3].y << "\n";
////	}

//	m_vertCount = verts.size();
//	m_bvhNodeCount = bvhData.size();
//	m_triIdxCount = triIndices.size();
////	exit(0);
}

void vRendererCuda::validateCuda(cudaError_t _err, const std::string &_msg)
{
  if(_err != cudaSuccess)
  {
		std::cerr << "Failed to perform a cuda operation: " << _msg << "\n";
    exit(0);
  }
}
