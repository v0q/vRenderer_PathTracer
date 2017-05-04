///
/// \file vRendererCuda.cpp
/// \brief Implements the Cuda version of the renderer
///

#include "vRendererCuda.h"

#include <chrono>
#include <cuda_gl_interop.h>

#include "Utilities.cuh"

vRendererCuda::vRendererCuda() :
	m_vertices(nullptr),
	m_normals(nullptr),
	m_tangents(nullptr),
	m_bvhData(nullptr),
	m_uvs(nullptr),
	m_hdr(nullptr),
	m_diffuse(nullptr),
	m_normal(nullptr),
	m_specular(nullptr),
	m_brdf(nullptr),
	m_frame(1),
	m_initialised(false)
{
	m_fresnelCoef = 0.1f;
	m_fresnelPow = 3.f;
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

	validateCuda(cudaMalloc(&m_vertices, 1), "Init vertex device pointer");
	validateCuda(cudaMalloc(&m_normals, 1), "Init normals device pointer");
  validateCuda(cudaMalloc(&m_bvhData, 1), "Init BVH node device pointer");

	cu_fillFloat4(m_colorArray, make_float4(0.0f, 0.0f, 0.0f, 0.0f), sz);
	validateCuda(cudaDeviceSynchronize(), "Init");

  m_initialised = true;
}

void vRendererCuda::registerTextureBuffer(GLuint &_texture)
{
  assert(m_initialised);
	validateCuda(cudaGraphicsGLRegisterImage(&m_cudaGLTextureBuffer, _texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
}

void vRendererCuda::registerDepthBuffer(GLuint &_depthTexture)
{
	assert(m_initialised);
	validateCuda(cudaGraphicsGLRegisterImage(&m_cudaGLDepthBuffer, _depthTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
}

void vRendererCuda::updateCamera()
{
	m_virtualCamera->consume();

	m_camera.m_origin.x = m_virtualCamera->getOrig().m_x;
	m_camera.m_origin.y = m_virtualCamera->getOrig().m_y;
	m_camera.m_origin.z = m_virtualCamera->getOrig().m_z;
	m_camera.m_origin.w = 0.f;

	m_camera.m_dir.x = m_virtualCamera->getDir().m_x;
	m_camera.m_dir.y = m_virtualCamera->getDir().m_y;
	m_camera.m_dir.z = m_virtualCamera->getDir().m_z;
	m_camera.m_dir.w = 0.f;

	m_camera.m_upV.x = m_virtualCamera->getUp().m_x;
	m_camera.m_upV.y = m_virtualCamera->getUp().m_y;
	m_camera.m_upV.z = m_virtualCamera->getUp().m_z;
	m_camera.m_upV.w = 0.f;

	m_camera.m_rightV.x = m_virtualCamera->getRight().m_x;
	m_camera.m_rightV.y = m_virtualCamera->getRight().m_y;
	m_camera.m_rightV.z = m_virtualCamera->getRight().m_z;
	m_camera.m_rightV.w = 0.f;

	m_camera.m_fovScale = m_virtualCamera->getFovScale();

	m_frame = 1;
	cu_fillFloat4(m_colorArray, make_float4(0.0f, 0.0f, 0.0f, 0.0f), m_width*m_height);
	validateCuda(cudaDeviceSynchronize(), "Update camera");
}

void vRendererCuda::clearBuffer()
{
	m_frame = 1;
	cu_fillFloat4(m_colorArray, make_float4(0.0f, 0.0f, 0.0f, 0.0f), m_width*m_height);
	validateCuda(cudaDeviceSynchronize(), "Clear buffer");
}

void vRendererCuda::render()
{
	if(m_virtualCamera->isDirty())
	{
		updateCamera();
	}

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	// Map the texture and depth buffers from OpenGL
	validateCuda(cudaGraphicsMapResources(1, &m_cudaGLTextureBuffer), "Map GL texture buffer");
	validateCuda(cudaGraphicsSubResourceGetMappedArray(&m_cudaImgArray, m_cudaGLTextureBuffer, 0, 0), "Attach the mapped texture buffer to a cuda resource");

	validateCuda(cudaGraphicsMapResources(1, &m_cudaGLDepthBuffer), "Map GL depth buffer");
	validateCuda(cudaGraphicsSubResourceGetMappedArray(&m_cudaDepthArray, m_cudaGLDepthBuffer, 0, 0), "Attach the mapped depth buffer to a cuda resource");

  cudaResourceDesc wdsc;
  wdsc.resType = cudaResourceTypeArray;
  wdsc.res.array.array = m_cudaImgArray;

	// Create a cuda writeable surface for the texture
	cudaSurfaceObject_t textureSurface;
	validateCuda(cudaCreateSurfaceObject(&textureSurface, &wdsc), "Create a writeable cuda surface");

	cudaResourceDesc wdscd;
	wdscd.resType = cudaResourceTypeArray;
	wdscd.res.array.array = m_cudaDepthArray;

	// Create a cuda writeable surface for the depth
	cudaSurfaceObject_t depthSurface;
	validateCuda(cudaCreateSurfaceObject(&depthSurface, &wdscd), "Create a writeable cuda surface");

	// Main render call
	cu_runRenderKernel(textureSurface,
										 depthSurface,
										 m_hdr,
										 m_vertices,
										 m_normals,
										 m_tangents,
										 m_bvhData,
										 m_uvs,
										 m_colorArray,
										 m_camera,
										 m_width,
										 m_height,
										 m_frame++,
										 std::chrono::duration_cast<std::chrono::milliseconds>(t1.time_since_epoch()).count(),
										 m_fresnelCoef,
										 m_fresnelPow);

	// Clean up and free the OpenGL buffers
	validateCuda(cudaDestroySurfaceObject(textureSurface), "Clean up the surface");
	validateCuda(cudaGraphicsUnmapResources(1, &m_cudaGLTextureBuffer), "Free the mapped GL texture buffer");

	validateCuda(cudaDestroySurfaceObject(depthSurface), "Clean up the surface");
	validateCuda(cudaGraphicsUnmapResources(1, &m_cudaGLDepthBuffer), "Free the mapped GL depth buffer");

	validateCuda(cudaStreamSynchronize(0), "Synchronize");
}

void vRendererCuda::cleanUp()
{
  if(m_initialised)
	{
		if(m_vertices)
			validateCuda(cudaFree(m_vertices), "Clean up triangle data buffer");
		if(m_tangents)
			validateCuda(cudaFree(m_tangents), "Clean up tangent buffer");
		if(m_normals)
      validateCuda(cudaFree(m_normals), "Clean up normals buffer");
		if(m_bvhData)
			validateCuda(cudaFree(m_bvhData), "Clean up BVH node buffer");
		if(m_uvs)
			validateCuda(cudaFree(m_uvs), "Clean up UV buffer");

		if(m_hdr)
			validateCuda(cudaFree(m_hdr), "Clean up HDR buffer");
		if(m_diffuse)
			validateCuda(cudaFree(m_diffuse), "Clean up diffuse memory");
		if(m_normal)
			validateCuda(cudaFree(m_normal), "Clean up normal memory");
		if(m_specular)
			validateCuda(cudaFree(m_specular), "Clean up specular memory");
		if(m_brdf)
			validateCuda(cudaFree(m_brdf), "Clean up BRDF buffer");

		validateCuda(cudaFree(m_colorArray), "Clean up color buffer");
		validateCuda(cudaGraphicsUnregisterResource(m_cudaGLTextureBuffer), "Unregister GL Texture");
		validateCuda(cudaGraphicsUnregisterResource(m_cudaGLDepthBuffer), "Unregister GL Depth texture");

		cu_cleanUp();
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
	std::vector<float4> tangents;
  std::vector<float2> uvs;

	bvhData.resize(4);

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
				for(unsigned int j = leaf->firstIndex(); j < leaf->lastIndex(); ++j)
				{
					unsigned int triInd = _meshData.m_bvh.getTriIndex(j);
					const vHTriangle tri = _meshData.m_triangles[triInd];
					// Get the triangle data and push it to their respective buffers
					for(unsigned int k = 0; k < 3; ++k)
					{
						const ngl::Vec3 &vert = _meshData.m_vertices[tri.m_indices[k]].m_vert;
						const ngl::Vec3 &tangent = _meshData.m_vertices[tri.m_indices[k]].m_tangent;
						const ngl::Vec3 &norm = _meshData.m_vertices[tri.m_indices[k]].m_normal;//_meshData.m_triangles[triInd].m_normal;

						uvs.push_back(make_float2(_meshData.m_vertices[tri.m_indices[k]].m_u, _meshData.m_vertices[tri.m_indices[k]].m_v));

						verts.push_back(make_float4(vert.m_x, vert.m_y, vert.m_z, 0.f));
						tangents.push_back(make_float4(tangent.m_x, tangent.m_y, tangent.m_z, 0.f));
						normals.push_back(make_float4(norm.m_x, norm.m_y, norm.m_z, 0.f));
          }
				}
				// Terminate triangles, doing this for normals and uvs as well to keep the indices matching
				verts.push_back(make_float4(intAsFloat(0x80000000), 0, 0, 0));
				tangents.push_back(make_float4(intAsFloat(0x80000000), 0, 0, 0));
				normals.push_back(make_float4(intAsFloat(0x80000000), 0, 0, 0));
				uvs.push_back(make_float2(intAsFloat(0x80000000), 0));
			}

		// Node bounding box
		// Stored int child 1 XY, child 2 XY, child 1 & 2 Z
		bvhData[idx + 0] = make_float4(bounds[0].minBounds().m_x, bounds[0].maxBounds().m_x, bounds[0].minBounds().m_y, bounds[0].maxBounds().m_y);
		bvhData[idx + 1] = make_float4(bounds[1].minBounds().m_x, bounds[1].maxBounds().m_x, bounds[1].minBounds().m_y, bounds[1].maxBounds().m_y);
		bvhData[idx + 2] = make_float4(bounds[0].minBounds().m_z, bounds[0].maxBounds().m_z, bounds[1].minBounds().m_z, bounds[1].maxBounds().m_z);

		// Storing indices as floats
		bvhData[idx + 3] = make_float4(intAsFloat(indices[0]), intAsFloat(indices[1]), 0, 0);
	}

	// Cleanup old device buffers and allocate new ones
	if(m_vertices)
	{
		validateCuda(cudaFree(m_vertices), "Delete old vertex buffer");
	}
	validateCuda(cudaMalloc(&m_vertices, verts.size()*sizeof(float4)), "Malloc vertex device pointer");
	validateCuda(cudaMemcpy(m_vertices, &verts[0], verts.size()*sizeof(float4), cudaMemcpyHostToDevice), "Copy vertex data to gpu");

	if(m_tangents)
	{
		validateCuda(cudaFree(m_tangents), "Delete old tangent buffer");
	}
	validateCuda(cudaMalloc(&m_tangents, tangents.size()*sizeof(float4)), "Malloc tangent device pointer");
	validateCuda(cudaMemcpy(m_tangents, &tangents[0], tangents.size()*sizeof(float4), cudaMemcpyHostToDevice), "Copy tangent data to gpu");

	if(m_uvs)
	{
		validateCuda(cudaFree(m_uvs), "Delete old uv buffer");
	}
	validateCuda(cudaMalloc(&m_uvs, uvs.size()*sizeof(float2)), "Malloc uv device pointer");
	validateCuda(cudaMemcpy(m_uvs, &uvs[0], uvs.size()*sizeof(float2), cudaMemcpyHostToDevice), "Copy uv data to gpu");

	if(m_normals)
	{
		validateCuda(cudaFree(m_normals), "Delete old normal buffer");
	}
	validateCuda(cudaMalloc(&m_normals, normals.size()*sizeof(float4)), "Malloc normals device pointer");
	validateCuda(cudaMemcpy(m_normals, &normals[0], normals.size()*sizeof(float4), cudaMemcpyHostToDevice), "Copy normal data to gpu");

	if(m_bvhData)
	{
		validateCuda(cudaFree(m_bvhData), "Delete old bvh node buffer");
	}
	validateCuda(cudaMalloc(&m_bvhData, bvhData.size()*sizeof(float4)), "Malloc BVH node device pointer");
  validateCuda(cudaMemcpy(m_bvhData, &bvhData[0], bvhData.size()*sizeof(float4), cudaMemcpyHostToDevice), "Copy bvh node data to gpu");

	cu_meshInitialised();
}

void vRendererCuda::loadHDR(const Imf::Rgba *_colours, const unsigned int &_w, const unsigned int &_h)
{
	float4 *dataAsFloats = new float4[_w*_h];

	for(unsigned int i = 0; i < _w*_h; ++i)
	{
		dataAsFloats[i] = make_float4(_colours[i].r, _colours[i].g, _colours[i].b, _colours[i].a);
	}

	if(m_hdr)
	{
		validateCuda(cudaFree(m_hdr), "Delete old HDR memory");
	}

	validateCuda(cudaMalloc(&m_hdr, _w*_h*sizeof(float4)), "Malloc HDR map device pointer");
	validateCuda(cudaMemcpy(m_hdr, dataAsFloats, _w*_h*sizeof(float4), cudaMemcpyHostToDevice), "Copy HDR map to gpu");

	cu_setHDRDim(_w, _h);

	delete [] dataAsFloats;
}

void vRendererCuda::loadTexture(const QImage &_texture, const float &_gamma, const unsigned int &_type)
{
	unsigned int w = _texture.width();
	unsigned int h = _texture.height();
	float correction = (_gamma > 0.001f ? 1.f/_gamma : 1.f);

	float4 *dataAsFloats = new float4[w*h];

	unsigned int k = 0;
	for(unsigned int j = 0; j < h; ++j)
	{
		for(unsigned int i = 0; i < w; ++i)
		{
			QColor pixel(_texture.pixel(i, j));
			if(_type == DIFFUSE)
			{
				// Invert gamma correction if it's already applied
				dataAsFloats[k++] = make_float4(std::pow(pixel.red()/255.f, correction),
																				std::pow(pixel.green()/255.f, correction),
																				std::pow(pixel.blue()/255.f, correction),
																				pixel.alpha()/255.f);
			}
			else
			{
				dataAsFloats[k++] = make_float4(pixel.red()/255.f, pixel.green()/255.f, pixel.blue()/255.f, pixel.alpha()/255.f);
			}
		}
	}

	// Load the texture data to the correct buffer
	switch(_type)
	{
		case DIFFUSE:
		{
			if(m_diffuse)
			{
				validateCuda(cudaFree(m_diffuse), "Delete old diffuse memory");
			}

			validateCuda(cudaMalloc(&m_diffuse, w*h*sizeof(float4)), "Diffuse texture memory allocation");
			validateCuda(cudaMemcpy(m_diffuse, dataAsFloats, w * h * sizeof(float4), cudaMemcpyHostToDevice), "Memcpy to diffuse texture");
			cu_bindTexture(m_diffuse, w, h, static_cast<vTextureType>(_type));
		} break;
		case NORMAL:
		{
			if(m_normal)
			{
				validateCuda(cudaFree(m_normal), "Delete old normal memory");
			}

			validateCuda(cudaMalloc(&m_normal, w*h*sizeof(float4)), "Normal texture memory allocation");
			validateCuda(cudaMemcpy(m_normal, dataAsFloats, w * h * sizeof(float4), cudaMemcpyHostToDevice), "Memcpy to normal texture");
			cu_bindTexture(m_normal, w, h, static_cast<vTextureType>(_type));
		} break;
		case SPECULAR:
		{
			if(m_specular)
			{
				validateCuda(cudaFree(m_specular), "Delete old specular memory");
			}

			validateCuda(cudaMalloc(&m_specular, w*h*sizeof(float4)), "Specular texture memory allocation");
			validateCuda(cudaMemcpy(m_specular, dataAsFloats, w * h * sizeof(float4), cudaMemcpyHostToDevice), "Memcpy to specular texture");
			cu_bindTexture(m_specular, w, h, static_cast<vTextureType>(_type));
		} break;
		default: break;
	}

	delete [] dataAsFloats;
}

bool vRendererCuda::loadBRDF(const float *_brdf)
{
	// If the brdf data is valid (e.g. not a nullptr), load it to the GPU
	if(_brdf)
	{
		if(m_brdf)
		{
			validateCuda(cudaFree(m_brdf), "Delete old brdf memory");
		}

		unsigned int n = BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D * BRDF_SAMPLING_RES_PHI_D / 2;

		// Copy brdf to GPU
		validateCuda(cudaMalloc(&m_brdf, n * 3 * sizeof(float)), "BRDF texture memory allocation");
		validateCuda(cudaMemcpy(m_brdf, _brdf, n * 3 * sizeof(float), cudaMemcpyHostToDevice), "Memcpy to brdf texture");
		cu_bindBRDF(m_brdf);

		delete [] _brdf;
		return true;
	}
	else
	{
		return false;
	}
}

void vRendererCuda::useExampleSphere(const bool &_newVal)
{
	cu_useExampleSphere(_newVal);
}

void vRendererCuda::useBRDF(const bool &_newVal)
{
	cu_useBRDF(_newVal);
}

void vRendererCuda::useCornellBox(const bool &_newVal)
{
	cu_useCornellBox(_newVal);
}

void vRendererCuda::validateCuda(cudaError_t _err, const std::string &_msg)
{
  if(_err != cudaSuccess)
  {
		std::cerr << "Failed to perform a cuda operation: " << _msg << "\n";
		std::cerr << "Err: " << cudaGetErrorString(_err) << "\n";

		FILE *log = fopen("errorlog.txt", "w");
		fprintf(log, "%s\n", cudaGetErrorString(_err));

		std::cerr << "Check errorlog.txt for more details\n";
    exit(0);
  }
}
