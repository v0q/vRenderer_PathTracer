#include "vRendererCuda.h"

#include <chrono>
#include <cuda_gl_interop.h>

vRendererCuda::vRendererCuda() :
  m_frame(1),
	m_triCount(0),
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
	validateCuda(cudaMalloc(&m_colorArray, sizeof(float4)*sz));
	validateCuda(cudaMalloc(&m_camera, sizeof(float4)));
	validateCuda(cudaMalloc(&m_camdir, sizeof(float4)));

	float4 cam = make_float4(50.0f, 52.0f, 295.6f, 0.0f);
	float4 camdir = make_float4(0.0f, -0.0425734f, -0.999093f, 0.0f);

	validateCuda(cudaMemcpy(m_camera, &cam, sizeof(float4), cudaMemcpyHostToDevice));
	validateCuda(cudaMemcpy(m_camdir, &camdir, sizeof(float4), cudaMemcpyHostToDevice));
	cu_fillFloat4(m_colorArray, make_float4(0.0f, 0.0f, 0.0f, 0.0f), sz);
	cudaDeviceSynchronize();

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
		validateCuda(cudaMemcpy(m_camera, _cam, sizeof(float4), cudaMemcpyHostToDevice));
	}
	if(_dir != nullptr)
	{
		validateCuda(cudaMemcpy(m_camdir, _dir, sizeof(float4), cudaMemcpyHostToDevice));
	}

	m_frame = 1;
	cu_fillFloat4(m_colorArray, make_float4(0.0f, 0.0f, 0.0f, 0.0f), m_width*m_height);
	cudaDeviceSynchronize();
}

void vRendererCuda::render()
{
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

  validateCuda(cudaGraphicsMapResources(1, &m_cudaGLTextureBuffer));
  validateCuda(cudaGraphicsSubResourceGetMappedArray(&m_cudaImgArray, m_cudaGLTextureBuffer, 0, 0));

  cudaResourceDesc wdsc;
  wdsc.resType = cudaResourceTypeArray;
  wdsc.res.array.array = m_cudaImgArray;
  cudaSurfaceObject_t writeSurface;
  validateCuda(cudaCreateSurfaceObject(&writeSurface, &wdsc));
	cu_runRenderKernel(writeSurface,
										 m_meshes[0],
										 m_colorArray,
										 m_camera,
										 m_camdir,
										 m_width, m_height, m_frame++, std::chrono::duration_cast<std::chrono::milliseconds>(t1.time_since_epoch()).count());
  validateCuda(cudaDestroySurfaceObject(writeSurface));
  validateCuda(cudaGraphicsUnmapResources(1, &m_cudaGLTextureBuffer));
  validateCuda(cudaStreamSynchronize(0));
}

void vRendererCuda::cleanUp()
{
  if(m_initialised)
  {
		for(auto &buffer : m_meshes)
			cudaFree(buffer);
    cudaFree(m_colorArray);
    cudaFree(m_camera);
    cudaFree(m_camdir);
    cudaGraphicsUnregisterResource(m_cudaGLTextureBuffer);
  }
}

void vRendererCuda::initMesh(const std::vector<vFloat3> &_vertData)
{
	unsigned int sz = _vertData.size();

	m_triCount = (sz - 2)/6;
	vMesh mesh;
	vTriangle *triangles = new vTriangle[m_triCount];

	// Get the vertices of the mesh and their normals
	for(unsigned int i = 0; i < sz - 2; i += 6)
	{
		vVert v1, v2, v3;
		v1.m_vert.x = _vertData[i].x;
		v1.m_vert.y = _vertData[i].y;
		v1.m_vert.z = _vertData[i].z;
		v1.m_vert.w = 0.f;
		v1.m_normal.x = _vertData[i+1].x;
		v1.m_normal.y = _vertData[i+1].y;
		v1.m_normal.z = _vertData[i+1].z;
		v1.m_normal.w = 0.f;
		v2.m_vert.x = _vertData[i+2].x;
		v2.m_vert.y = _vertData[i+2].y;
		v2.m_vert.z = _vertData[i+2].z;
		v2.m_vert.w = 0.f;
		v2.m_normal.x = _vertData[i+3].x;
		v2.m_normal.y = _vertData[i+3].y;
		v2.m_normal.z = _vertData[i+3].z;
		v2.m_normal.w = 0.f;
		v3.m_vert.x = _vertData[i+4].x;
		v3.m_vert.y = _vertData[i+4].y;
		v3.m_vert.z = _vertData[i+4].z;
		v3.m_vert.w = 0.f;
		v3.m_normal.x = _vertData[i+5].x;
		v3.m_normal.y = _vertData[i+5].y;
		v3.m_normal.z = _vertData[i+5].z;
		v3.m_normal.w = 0.f;
		triangles[i/6].m_v1 = v1;
		triangles[i/6].m_v2 = v2;
		triangles[i/6].m_v3 = v3;
	}
	// Set the triangle count and bounding box of the mesh
	mesh.m_bb.m_x = make_float2(_vertData[sz - 2].x, _vertData[sz - 1].x);
	mesh.m_bb.m_y = make_float2(_vertData[sz - 2].y, _vertData[sz - 1].y);
	mesh.m_bb.m_z = make_float2(_vertData[sz - 2].z, _vertData[sz - 1].z);
	mesh.m_triCount = m_triCount;

	vMesh *meshBuffer;
	vTriangle *triangleBuffer;

	validateCuda(cudaMalloc(&meshBuffer, sizeof(vMesh)), "Malloc mesh buffer");
	validateCuda(cudaMalloc(&triangleBuffer, m_triCount*sizeof(vTriangle)), "Malloc triangle buffer");
	validateCuda(cudaMemcpy(meshBuffer, &mesh, sizeof(vMesh), cudaMemcpyHostToDevice), "Copy mesh to host");
	validateCuda(cudaMemcpy(triangleBuffer, &triangles[0], m_triCount*sizeof(vTriangle), cudaMemcpyHostToDevice), "Copy triangles to host");
	validateCuda(cudaMemcpy(&(meshBuffer->m_mesh), &triangleBuffer, sizeof(vTriangle *), cudaMemcpyHostToDevice), "Copy triangles host to mesh host");

	m_meshes.push_back(meshBuffer);

	delete [] triangles;
}

void vRendererCuda::validateCuda(cudaError_t _err, const std::string &_msg)
{
  if(_err != cudaSuccess)
  {
		std::cerr << "Failed to perform a cuda operation: " << _msg << "\n";
    exit(0);
  }
}
