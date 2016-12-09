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
	return make_float3(_a.y * _b.z - _a.z * _b.y, _a.z * _b.x - _a.x * _b.z, _a.x * _b.y - _a.y * _b.z);
}

inline __device__ float3 normalize(const float3 &_a)
{
	return _a*(1.0 / sqrtf(_a.x*_a.x + _a.y*_a.y + _a.z*_a.z));
}

inline __device__ float clamp(const float &_val, const float &_low, const float &_hi)
{
	return _val < _low ? _low : (_val > _hi ? _hi : _val);
}

inline __device__ float3 clamp(const float3 &_val, const float &_low, const float &_hi)
{
	float x =_val.x < _low ? _low : (_val.x > _hi ? _hi : _val.x);
	float y =_val.y < _low ? _low : (_val.y > _hi ? _hi : _val.y);
	float z =_val.z < _low ? _low : (_val.z > _hi ? _hi : _val.z);
	return make_float3(x, y, z);
}
