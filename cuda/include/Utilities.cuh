#pragma once

#include <cuda_runtime.h>

#define BRDF_SAMPLING_RES_THETA_H       90
#define BRDF_SAMPLING_RES_THETA_D       90
#define BRDF_SAMPLING_RES_PHI_D         360

inline __device__ __host__ float intAsFloat(const int &_val)
{
	union
	{
		float a;
		int b;
	} t;
	t.b = _val;

	return t.a;
}

inline __device__ __host__ float floatAsInt(const float &_val)
{
	union
	{
		float a;
		int b;
	} t;
	t.a = _val;

	return t.b;
}

inline __device__ void swap(int &_a, int &_b)
{
	int tmp = _a;
	_a = _b;
	_b = tmp;
}
