#pragma once

#include <GL/glew.h>
#include <string>

#ifdef __VRENDERER_OPENCL__
	#ifdef __APPLE__
		#include <OpenCL/cl.h>
	#else
		#include <CL/cl.h>
	#endif
	typedef cl_float4 ftype;
#elif __VRENDERER_CUDA__
	#include <cuda/cuda_runtime.h>
	typedef float4 ftype;
#endif

typedef struct vFloat3
{
  float x;
  float y;
  float z;
	vFloat3(const float &_x, const float &_y, const float &_z) : x(_x), y(_y), z(_z) {}
} vFloat3;

class vMeshLoader
{
public:
	vMeshLoader(const std::string &_mesh);
	~vMeshLoader();
	static std::vector<vFloat3> loadMesh(const std::string &_mesh);
private:
};
