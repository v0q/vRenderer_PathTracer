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
//	cu_runRenderKernel(writeSurface,
//										 m_triangleData,
//										 m_triIdxList,
//										 m_bvhLimits,
//										 m_bvhChildrenOrTriangles,
//										 m_triCount,
//										 m_bvhBoxCount,
//										 m_triIdxCount,
//										 m_colorArray,
//										 m_camera,
//										 m_camdir,
//										 m_width,
//										 m_height,
//										 m_frame++,
//										 std::chrono::duration_cast<std::chrono::milliseconds>(t1.time_since_epoch()).count());
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

void vRendererCuda::initMesh(const SBVH &_sbvhData)
{
	std::vector<const SBVHNode *> stack{_sbvhData.root()};
	std::vector<unsigned int> triIndices;
	std::vector<ngl::Vec3> verts;
	while(stack.size())
	{
		const SBVHNode *node = stack.back();
		stack.pop_back();

		AABB bounds[2];
//		int indices[2];

		for(unsigned int i = 0; i < 2; ++i)
		{
			const SBVHNode *child = node->childNode(i);
			bounds[i] = child->getBounds();

			if(!child->isLeaf())
			{
				stack.push_back(child);
				continue;
//				indices[i] =
			}

			const LeafNode *leaf = dynamic_cast<const LeafNode *>(child);

			for(unsigned int j = leaf->firstIndex(); j < leaf->lastIndex(); ++j)
			{
				for(unsigned int k = 0; k < 3; ++k)
				{
					const ngl::Vec3 &vert = _sbvhData.getVert(_sbvhData.getTriangle(j).m_indices[0]);
					verts.push_back(vert);
				}
				triIndices.push_back(_sbvhData.getTriIndex(j));
			}
		}
	}
}

void vRendererCuda::validateCuda(cudaError_t _err, const std::string &_msg)
{
  if(_err != cudaSuccess)
  {
		std::cerr << "Failed to perform a cuda operation: " << _msg << "\n";
    exit(0);
  }
}
