#pragma once

#include <cuda_runtime.h>

inline __device__ float3 operator+(const float3 &_a, const float3 &_b)
{
	return make_float3(_a.x + _b.x, _a.y + _b.y, _a.z + _b.z);
}

inline __device__ float3 operator-(const float3 &_a, const float3 &_b)
{
	return make_float3(_a.x - _b.x, _a.y - _b.y, _a.z - _b.z);
}

inline __device__ void operator+=(float3 &_a, const float3 &_b)
{
	_a.x += _b.x;
	_a.y += _b.y;
	_a.z += _b.z;
}

inline __device__ void operator*=(float3 &_a, const float3 &_b)
{
	_a.x *= _b.x;
	_a.y *= _b.y;
	_a.z *= _b.z;
}

inline __device__ void operator*=(float3 &_a, const float &_b)
{
	_a.x *= _b;
	_a.y *= _b;
	_a.z *= _b;
}

inline __device__ void operator/=(float3 &_a, const float &_b)
{
	_a.x /= _b;
	_a.y /= _b;
	_a.z /= _b;
}

inline __device__ float3 operator*(const float3 &_a, const float3 &_b)
{
	return make_float3(_a.x*_b.x, _a.y*_b.y, _a.z*_b.z);
}

inline __device__ float3 operator*(const float3 &_a, const float &_b)
{
	return make_float3(_a.x*_b, _a.y*_b, _a.z*_b);
}

inline __device__ float3 operator*(const float &_a, const float3 &_b)
{
	return make_float3(_a*_b.x, _a*_b.y, _a*_b.z);
}

inline __device__ float dot(const float3 &_a, const float3 &_b)
{
	return _a.x*_b.x + _a.y*_b.y + _a.z*_b.z;
}

inline __device__ float3 cross(const float3 &_a, const float3 &_b)
{
	return make_float3(_a.y * _b.z - _a.z * _b.y, _a.z * _b.x - _a.x * _b.z, _a.x * _b.y - _a.y * _b.x);
}

inline __device__ float3 normalize(const float3 &_a)
{
	float invLen = rsqrtf(dot(_a, _a));
	return _a * invLen;
}

inline __device__ float clamp(const float &_val, const float &_low, const float &_hi)
{
	return max(_low, min(_hi, _val));
}

inline __device__ float3 clamp(const float3 &_val, const float &_low, const float &_hi)
{
	float x = clamp(_val.x, _low, _hi);
	float y = clamp(_val.y, _low, _hi);
	float z = clamp(_val.z, _low, _hi);
	return make_float3(x, y, z);
}
