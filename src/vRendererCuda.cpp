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
	float4 camdir = make_float4(-0.164325f, -0.170898f, -0.971489f, 0.0f);

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
										 m_normals,
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
		if(m_normals)
			validateCuda(cudaFree(m_normals), "Clean up normals buffer");
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

void vRendererCuda::initMesh(const vMeshData &_meshData)
{
	// Create a stack for node paired with an index
	std::vector<std::pair<const BVHNode *, unsigned int>> nodeStack{std::make_pair(_meshData.m_bvh.getRoot(), 0)};

	// Vector for bvh node data: child node/triangle indices, aabb's
	std::vector<float4> bvhData;
	std::vector<float4> verts;
	std::vector<float4> normals;
	std::vector<unsigned int> triIndices;

	bvhData.resize(4);
	unsigned int lowestVertIdx = INT_MAX;
	unsigned int highestVertIdx = 0;

	while(nodeStack.size())
	{
		const BVHNode *node = nodeStack.back().first;
		unsigned int idx = nodeStack.back().second;
		nodeStack.pop_back();

		AABB bounds[2];
		int indices[2];
//		if(!node->isLeaf())
//		{
			for(unsigned int i = 0; i < 2; ++i)
			{
				// Get the bounds of the node
				const BVHNode *child = node->childNode(i);
				bounds[i] = child->getBounds();

				if(!child->isLeaf())
				{
					// Index for the next node is an offset in memory
					unsigned int cidx = bvhData.size();
					indices[i] = cidx;// * sizeof(float4);
					nodeStack.push_back(std::make_pair(child, cidx));

					// Allocate space for the node data (e.g. bounds, indices etc)
					bvhData.resize(bvhData.size() + 4);
					continue;
				}

				const LeafNode *leaf = dynamic_cast<const LeafNode *>(child);
				// Triangle index stored as its complement to distinquish them from child nodes (e.g. ~0 = -1, ~1 = -2...)
				indices[i] = ~verts.size();
				std::cout << "Leaf index: " << verts.size() << "\n";
				std::cout << "First index: " << leaf->firstIndex() << ", Last index: " << leaf->lastIndex() << "\n";
				std::cout << "Next index: " << ((leaf->lastIndex()  - leaf->firstIndex())*3 + 1)+verts.size() << "\n\n";
				for(unsigned int j = leaf->firstIndex(); j < leaf->lastIndex(); ++j)
				{
					unsigned int triInd = _meshData.m_bvh.getTriIndex(j);
					for(unsigned int k = 0; k < 3; ++k)
					{
						const ngl::Vec3 &vert = _meshData.m_vertices[_meshData.m_triangles[triInd].m_indices[k]];
						const ngl::Vec3 &norm = _meshData.m_triangles[triInd].m_normal;
						verts.push_back(make_float4(vert.m_x, vert.m_y, vert.m_z, 0.f));
						normals.push_back(make_float4(norm.m_x, norm.m_y, norm.m_z, 0.f));
					}
					triIndices.push_back(triInd);
				}
				// Terminate triangles
				verts.push_back(make_float4(intAsFloat(0x80000000), 0, 0, 0));
				normals.push_back(make_float4(intAsFloat(0x80000000), 0, 0, 0));
			}
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

		// Node bounding box
		// Stored int child 1 XY, child 2 XY, child 1 & 2 Z
		bvhData[idx + 0] = make_float4(bounds[0].minBounds().m_x, bounds[0].maxBounds().m_x, bounds[0].minBounds().m_y, bounds[0].maxBounds().m_y);
		bvhData[idx + 1] = make_float4(bounds[1].minBounds().m_x, bounds[1].maxBounds().m_x, bounds[1].minBounds().m_y, bounds[1].maxBounds().m_y);
		bvhData[idx + 2] = make_float4(bounds[0].minBounds().m_z, bounds[0].maxBounds().m_z, bounds[1].minBounds().m_z, bounds[1].maxBounds().m_z);

		// Storing indices as floats
		bvhData[idx + 3] = make_float4(intAsFloat(indices[0]), intAsFloat(indices[1]), 0, 0);
	}

	validateCuda(cudaMalloc(&m_vertices, verts.size()*sizeof(float4)), "Malloc vertex device pointer");
	validateCuda(cudaMemcpy(m_vertices, &verts[0], verts.size()*sizeof(float4), cudaMemcpyHostToDevice), "Copy vertex data to gpu");

	validateCuda(cudaMalloc(&m_normals, normals.size()*sizeof(float4)), "Malloc normals device pointer");
	validateCuda(cudaMemcpy(m_normals, &normals[0], normals.size()*sizeof(float4), cudaMemcpyHostToDevice), "Copy normal data to gpu");

	validateCuda(cudaMalloc(&m_bvhData, bvhData.size()*sizeof(float4)), "Malloc BVH node device pointer");
	validateCuda(cudaMemcpy(m_bvhData, &bvhData[0], bvhData.size()*sizeof(float4), cudaMemcpyHostToDevice), "Copy bvh node data to gpu");

	validateCuda(cudaMalloc(&m_triIdxList, triIndices.size()*sizeof(unsigned int)), "Malloc triangle index device pointer");
	validateCuda(cudaMemcpy(m_triIdxList, &triIndices[0], triIndices.size()*sizeof(unsigned int), cudaMemcpyHostToDevice), "Copy triangle indices to gpu");

	for(unsigned int i = 0; i < verts.size(); ++i)
	{
		if(*((int*)&verts[i].x) == 0x80000000)
			++i;
		if(i >= verts.size())
			break;
		std::cout << "v " << verts[i].x << " " << verts[i].y << " " << verts[i].z << "\n";
	}
	int ind = 1;
	for(unsigned int i = 0; i < verts.size(); i += 3)
	{
		if(*((int*)&verts[i].x) == 0x80000000)
			++i;
		if(i >= verts.size())
			break;
		std::cout << "f " << ind << "// " << ind + 1 << "// " << ind + 2 << "//\n";
		ind += 3;
	}

//	for(unsigned int i = 0; i < bvhData.size(); i += 4)
//	{
//		std::cout << "Index: " << i << "\n";
//		std::cout << "Child 1:\n";
//		std::cout << " BB: Min[" << bvhData[i + 0].x << ", " << bvhData[i + 0].z << ", " << bvhData[i + 2].x << "], Max[" << bvhData[i + 0].y << " " << bvhData[i + 0].w << ", " << bvhData[i + 2].y << "]\n";

//		std::cout << "Child 2:\n";
//		std::cout << " BB: Min[" << bvhData[i + 1].x << ", " << bvhData[i + 1].z << ", " << bvhData[i + 2].z << "], Max[" << bvhData[i + 1].y << " " << bvhData[i + 1].w << ", " << bvhData[i + 2].w << "]\n";

//		std::cout << "Indices " << *((int*)&bvhData[i + 3].x) << " " << *((int*)&bvhData[i + 3].y) << "\n\n";
//	}

	m_vertCount = verts.size();
	m_bvhNodeCount = bvhData.size();
	m_triIdxCount = triIndices.size();
//	exit(0);
}

void vRendererCuda::validateCuda(cudaError_t _err, const std::string &_msg)
{
  if(_err != cudaSuccess)
  {
		std::cerr << "Failed to perform a cuda operation: " << _msg << "\n";
    exit(0);
  }
}
