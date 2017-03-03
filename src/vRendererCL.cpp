/*
 * Ref: http://sa10.idav.ucdavis.edu/docs/sa10-dg-opencl-gl-interop.pdf
 */

#include <iostream>
#include <string>
#include <fstream>
#include <streambuf>
#include <vector>
#include <chrono>


#ifdef __APPLE__
  #include <GL/glew.h>
  #include <OpenCL/cl_gl_ext.h>
  #include <OpenGL.h>
#else
	#include <cuda/CL/cl_gl_ext.h>
	#include <GL/glew.h>
	#include <GL/glx.h>
#endif

#include "vRendererCL.h"

vRendererCL::vRendererCL() :
  m_frame(1),
  m_triCount(0),
  m_initialised(false)
{
  std::cout << "OpenCL vRenderer ctor called\n";
  m_GLBuffers.clear();
}

vRendererCL::~vRendererCL()
{
  std::cout << "OpenCL vRenderer dtor called\n";
  cleanUp();
}

void vRendererCL::init(const unsigned int &_w, const unsigned int &_h)
{
  assert(_w != 0 && _h != 0);
  m_width = _w;
	m_height = _h;

	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	std::cout << "Available platforms: \n";
	for(auto &p : platforms)
		std::cout << "\t" << p.getInfo<CL_PLATFORM_NAME>() << "\n";

	if(platforms.size())
	{
		std::cout << "Using platform: \n";
		m_platform = platforms[platforms.size() - 1];
		std::cout << "\t" << m_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
	}

	// Getting GPU devices
	std::vector<cl::Device> devices;
	m_platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

	std::cout << "Available devices: \n";
	for(auto &d : devices)
		std::cout << "\t" << d.getInfo<CL_DEVICE_NAME>() << "\n";

	if(devices.size())
	{
		std::cout << "Using device: \n";
    m_device = devices[devices.size() - 1];
		std::cout << "\t" << m_device.getInfo<CL_DEVICE_NAME>() << "\n";
	}

	std::cout << m_device.getInfo<CL_DEVICE_EXTENSIONS>() << "\n";

#ifdef __APPLE__
	CGLContextObj kCGLContext = CGLGetCurrentContext();
	CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
	cl_context_properties properties[3] =
	{
		CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
		(cl_context_properties)kCGLShareGroup,
		0
	};
#else
	cl_context_properties properties[7] =
	{
		CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
		CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
		CL_CONTEXT_PLATFORM, (cl_context_properties)(m_platform)(),
		0
	};
#endif

	m_context = cl::Context(m_device, properties);

	std::ifstream clFile("cl/src/PathTracer.cl");
	if(!clFile)
	{
		std::cerr << "Could not find 'cl/src/PathTracer.cl'\n";
		exit(0);
	}
	std::string pathTracerSrc((std::istreambuf_iterator<char>(clFile)),
														 std::istreambuf_iterator<char>());

	m_program = cl::Program(m_context, pathTracerSrc.c_str());
	cl_int result = m_program.build({ m_device }, "-cl-fast-relaxed-math");
	if(result)
	{
		std::cerr << "Failed compile the program: " << result << "\n";
		std::string buildlog = m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device);
		FILE *log = fopen("errorlog.txt", "w");
		fprintf(log, "%s\n", buildlog.c_str());

		std::cerr << "Build log saved to errorlog.txt:\n";
		exit(EXIT_FAILURE);
	}

  m_kernel = cl::Kernel(m_program, "render");

	m_queue = cl::CommandQueue(m_context, m_device);
	m_colorArray = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, m_width*m_height*sizeof(cl_float3));

  m_camera.x = 50.f;
  m_camera.y = 52.f;
  m_camera.z = 295.6f;
	m_camera.w = 0.f;

  m_camdir.x = 0.f;
	m_camdir.y = -0.0425734f;
	m_camdir.z = -0.999093f;
	m_camdir.w = 0.f;

  m_initialised = true;
}

void vRendererCL::registerTextureBuffer(GLuint &_texture)
{
	assert(m_initialised);
	cl_int err;
	m_glTexture = cl::ImageGL(m_context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, _texture, &err);
	if(err != CL_SUCCESS)
	{
		std::cout << "Failed to create OpenGL texture reference!" << err << "\n";
		exit(EXIT_FAILURE);
	}
	m_GLBuffers.push_back(m_glTexture);
}

void vRendererCL::render()
{
	cl::Event event;
	cl::NDRange globalRange = cl::NDRange(m_width, m_height);
	cl::NDRange localRange = cl::NDRange(16, 16);
	cl_int err;

	glFinish();
	static std::chrono::high_resolution_clock::time_point t0;
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
//  std::cout << "Rendering...\n";

	if((err = m_queue.enqueueAcquireGLObjects(&m_GLBuffers, nullptr, &event)) != CL_SUCCESS)
	{
		std::cout << "Failed to acquire gl objects: " << err << "\n";
		exit(EXIT_FAILURE);
  }

	event.wait();

  m_kernel.setArg(0, m_glTexture);
  m_kernel.setArg(1, m_vertices);
  m_kernel.setArg(2, m_normals);
  m_kernel.setArg(3, m_bvhNodes);
  m_kernel.setArg(4, m_triIdxList);
  m_kernel.setArg(5, m_colorArray);
  m_kernel.setArg(6, m_camera);
  m_kernel.setArg(7, m_camdir);
  m_kernel.setArg(8, m_width);
  m_kernel.setArg(9, m_height);
  m_kernel.setArg(10, m_frame++);
  m_kernel.setArg(11, static_cast<unsigned int>(std::chrono::duration_cast<std::chrono::milliseconds>(t1.time_since_epoch()).count()));

  if((err = m_queue.enqueueNDRangeKernel(m_kernel, cl::NullRange, globalRange, localRange, nullptr, &event)) != CL_SUCCESS)
  {
    std::cout << "Failed to enqueue the kernel: " << err << "\n";
    exit(EXIT_FAILURE);
  }
	event.wait();

	cl::finish();

	if((err = m_queue.enqueueReleaseGLObjects(&m_GLBuffers, nullptr, &event)) != CL_SUCCESS)
	{
		std::cout << "Failed to release gl objects: " << err << "\n";
		exit(EXIT_FAILURE);
	}
	event.wait();
	t0 = t1;
}

void vRendererCL::cleanUp()
{
  if(m_initialised)
  {
    m_GLBuffers.clear();
  }
}

void vRendererCL::updateCamera(const float *_cam, const float *_dir)
{
  if(_cam != nullptr)
  {
    m_camera.x = _cam[0];
    m_camera.y = _cam[1];
    m_camera.z = _cam[2];
		m_camera.w = 0.f;
  }
  if(_dir != nullptr)
  {
    m_camdir.x = _dir[0];
    m_camdir.y = _dir[1];
    m_camdir.z = _dir[2];
		m_camdir.w = 0.f;
  }

  m_frame = 1;
}

void vRendererCL::initMesh(const vMeshData &_meshData)
{
	cl_int err;
  // Create a stack for node paired with an index
  std::vector<std::pair<const BVHNode *, unsigned int>> nodeStack{std::make_pair(_meshData.m_bvh.getRoot(), 0)};

  // Vector for bvh node data: child node/triangle indices, aabb's
  std::vector<cl_float4> bvhData;
  std::vector<cl_float4> verts;
  std::vector<cl_float4> normals;
  std::vector<unsigned int> triIndices;

  bvhData.resize(4);

  cl_float4 terminator;
  terminator.x = intAsFloat(0x80000000);
  terminator.y = 0.f;
  terminator.z = 0.f;
  terminator.w = 0.f;

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
          for(unsigned int k = 0; k < 3; ++k)
          {
            const ngl::Vec3 &vert = _meshData.m_vertices[_meshData.m_triangles[triInd].m_indices[k]];
            const ngl::Vec3 &norm = _meshData.m_triangles[triInd].m_normal;
            cl_float4 v;
            cl_float4 n;
            v.x = vert.m_x;
            v.y = vert.m_y;
            v.z = vert.m_z;
            v.w = 0.f;
            n.x = norm.m_x;
            n.y = norm.m_y;
            n.z = norm.m_z;
            n.w = 0.f;
            verts.push_back(v);
            normals.push_back(n);
          }
          triIndices.push_back(triInd);
        }
        // Terminate triangles
        verts.push_back(terminator);
        normals.push_back(terminator);
      }

    // Node bounding box
    // Stored int child 1 XY, child 2 XY, child 1 & 2 Z
    bvhData[idx + 0].x = bounds[0].minBounds().m_x;
    bvhData[idx + 0].y = bounds[0].maxBounds().m_x;
    bvhData[idx + 0].z = bounds[0].minBounds().m_y;
    bvhData[idx + 0].w = bounds[0].maxBounds().m_y;

    bvhData[idx + 1].x = bounds[1].minBounds().m_x;
    bvhData[idx + 1].y = bounds[1].maxBounds().m_x;
    bvhData[idx + 1].z = bounds[1].minBounds().m_y;
    bvhData[idx + 1].w = bounds[1].maxBounds().m_y;

    bvhData[idx + 2].x = bounds[0].minBounds().m_z;
    bvhData[idx + 2].y = bounds[0].maxBounds().m_z;
    bvhData[idx + 2].z = bounds[1].minBounds().m_z;
    bvhData[idx + 2].w = bounds[1].maxBounds().m_z;

    // Storing indices as floats
    bvhData[idx + 3].x = intAsFloat(indices[0]);
    bvhData[idx + 3].y = intAsFloat(indices[1]);
    bvhData[idx + 3].y = 0.f;
    bvhData[idx + 3].w = 0.f;
  }

  cl::ImageFormat format;
  format.image_channel_order = CL_RGBA;
  format.image_channel_data_type = CL_FLOAT;
  // Copy buffers to GPU
  m_vertices = cl::Image1D(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, format, verts.size(), &verts[0], &err);

  if(err != CL_SUCCESS)
  {
    std::cerr << "Failed to copy vertex data to GPU " << err << "\n";
    exit(EXIT_FAILURE);
  }

  m_normals = cl::Image1D(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, format, normals.size(), &normals[0], &err);

  if(err != CL_SUCCESS)
  {
    std::cerr << "Failed to copy normal data to GPU " << err << "\n";
    exit(EXIT_FAILURE);
  }

  m_bvhNodes = cl::Image1D(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, format, bvhData.size(), &bvhData[0], &err);

  if(err != CL_SUCCESS)
  {
    std::cerr << "Failed to BVH data to GPU " << err << "\n";
    exit(EXIT_FAILURE);
  }

  cl::ImageFormat format2;
  format2.image_channel_order = CL_R;
  format2.image_channel_data_type = CL_UNSIGNED_INT32;
  m_triIdxList = cl::Image1D(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, format2, triIndices.size(), &triIndices[0], &err);

  if(err != CL_SUCCESS)
  {
    std::cerr << "Failed to copy triangle indices to GPU " << err << "\n";
    exit(EXIT_FAILURE);
  }

//	validateCuda(cudaMalloc(&m_vertices, verts.size()*sizeof(float4)), "Malloc vertex device pointer");
//	validateCuda(cudaMemcpy(m_vertices, &verts[0], verts.size()*sizeof(float4), cudaMemcpyHostToDevice), "Copy vertex data to gpu");

//	validateCuda(cudaMalloc(&m_normals, normals.size()*sizeof(float4)), "Malloc normals device pointer");
//	validateCuda(cudaMemcpy(m_normals, &normals[0], normals.size()*sizeof(float4), cudaMemcpyHostToDevice), "Copy normal data to gpu");

//	validateCuda(cudaMalloc(&m_bvhData, bvhData.size()*sizeof(float4)), "Malloc BVH node device pointer");
//	validateCuda(cudaMemcpy(m_bvhData, &bvhData[0], bvhData.size()*sizeof(float4), cudaMemcpyHostToDevice), "Copy bvh node data to gpu");

//	validateCuda(cudaMalloc(&m_triIdxList, triIndices.size()*sizeof(unsigned int)), "Malloc triangle index device pointer");
//	validateCuda(cudaMemcpy(m_triIdxList, &triIndices[0], triIndices.size()*sizeof(unsigned int), cudaMemcpyHostToDevice), "Copy triangle indices to gpu");

//  m_vertCount = verts.size();
//  m_bvhNodeCount = bvhData.size();
//  m_triIdxCount = triIndices.size();
//  // Triangle data
//  cl_float4 *triData = new cl_float4[_meshData.m_triangles.size() * 5];
//  for(unsigned int i = 0; i < _meshData.m_triangles.size(); ++i)
//  {
//    triData[5 * i	+ 0].x = _meshData.m_triangles[i].m_center.x;
//    triData[5 * i	+ 0].y = _meshData.m_triangles[i].m_center.y;
//    triData[5 * i	+ 0].z = _meshData.m_triangles[i].m_center.z;
//    triData[5 * i	+ 0].w = 0.f;

//    triData[5 * i + 1].x = _meshData.m_triangles[i].m_normal.x;
//    triData[5 * i + 1].y = _meshData.m_triangles[i].m_normal.y;
//    triData[5 * i + 1].z = _meshData.m_triangles[i].m_normal.z;
//    triData[5 * i + 1].w = _meshData.m_triangles[i].m_d;

//    triData[5 * i + 2].x = _meshData.m_triangles[i].m_e1.x;
//    triData[5 * i + 2].y = _meshData.m_triangles[i].m_e1.y;
//    triData[5 * i + 2].z = _meshData.m_triangles[i].m_e1.z;
//    triData[5 * i + 2].w = _meshData.m_triangles[i].m_d1;

//    triData[5 * i + 3].x = _meshData.m_triangles[i].m_e2.x;
//    triData[5 * i + 3].y = _meshData.m_triangles[i].m_e2.y;
//    triData[5 * i + 3].z = _meshData.m_triangles[i].m_e2.z;
//    triData[5 * i + 3].w = _meshData.m_triangles[i].m_d2;

//    triData[5 * i + 4].x = _meshData.m_triangles[i].m_e3.x;
//    triData[5 * i + 4].y = _meshData.m_triangles[i].m_e3.y;
//    triData[5 * i + 4].z = _meshData.m_triangles[i].m_e3.z;
//    triData[5 * i + 4].w = _meshData.m_triangles[i].m_d3;
//  }

//  cl::ImageFormat format;
//  format.image_channel_order = CL_RGBA;
//  format.image_channel_data_type = CL_FLOAT;
//  m_triangleData = cl::Image1D(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, format, _meshData.m_triangles.size() * 5, triData, &err);

//  std::cout << "Triangle data allocation and copy err: " << err << "\n";

//  m_triCount = _meshData.m_triangles.size();

//  delete [] triData;

//  m_triIdxCount = _meshData.m_cfbvhTriIndCount;

//  // BVH Limits
//  cl_float2 *bvhLimits = new cl_float2[_meshData.m_cfbvhBoxCount * 3];
//  for(unsigned int i = 0; i < _meshData.m_cfbvhBoxCount; ++i)
//  {
//    bvhLimits[3 * i + 0].x = _meshData.m_cfbvh[i].m_bottom.x;
//    bvhLimits[3 * i + 0].y = _meshData.m_cfbvh[i].m_top.x;

//    bvhLimits[3 * i + 1].x = _meshData.m_cfbvh[i].m_bottom.y;
//    bvhLimits[3 * i + 1].y = _meshData.m_cfbvh[i].m_top.z;

//    bvhLimits[3 * i + 2].x = _meshData.m_cfbvh[i].m_bottom.y;
//    bvhLimits[3 * i + 2].y = _meshData.m_cfbvh[i].m_top.z;
//  }

//  cl::ImageFormat format2;
//  format2.image_channel_order = CL_RG;
//  format2.image_channel_data_type = CL_FLOAT;
//  m_bvhLimits = cl::Image1D(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, format2, _meshData.m_cfbvhBoxCount * 3, bvhLimits, &err);
//  std::cout << "BVH limits allocation and copy err: " << err << "\n";

//  m_bvhBoxCount = _meshData.m_cfbvhBoxCount;

//  delete [] bvhLimits;

//  // Triangle indices

//  cl::ImageFormat format3;
//  format3.image_channel_order = CL_R;
//  format3.image_channel_data_type = CL_UNSIGNED_INT8;
//  m_triIdxList = cl::Image1D(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, format3, _meshData.m_cfbvhTriIndCount, _meshData.m_cfbvhTriIndices, &err);

//  std::cout << "Triangle index allocation and copy err: " << err << "\n";

//  // No need to have this and the limits in separate loops but makes it easier to follow
//  cl_uint4 *bvhChildrenOrTriangles = new cl_uint4[_meshData.m_cfbvhBoxCount];
//  for(unsigned int i = 0; i < _meshData.m_cfbvhBoxCount; ++i)
//  {
//    bvhChildrenOrTriangles[i].x = _meshData.m_cfbvh[i].m_u.m_leaf.m_count;
//    bvhChildrenOrTriangles[i].y = _meshData.m_cfbvh[i].m_u.m_inner.m_rightIndex;
//    bvhChildrenOrTriangles[i].z = _meshData.m_cfbvh[i].m_u.m_inner.m_leftIndex;
//    bvhChildrenOrTriangles[i].w = _meshData.m_cfbvh[i].m_u.m_leaf.m_startIndexInTriIndexList;
//  }


//  cl::ImageFormat format4;
//  format4.image_channel_order = CL_RGBA;
//  format4.image_channel_data_type = CL_UNSIGNED_INT8;
//  m_bvhChildrenOrTriangles = cl::Image1D(m_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, format4, _meshData.m_cfbvhBoxCount, bvhChildrenOrTriangles, &err);

//  std::cout << "BVH child nodes and triangles allocation and copy err: " << err << "\n";

//  delete [] bvhChildrenOrTriangles;
}

float vRendererCL::intAsFloat(const int &_v)
{
  union
  {
    int a;
    float b;
  } a;
  a.a = _v;

  return a.b;
}
