#pragma once

#include <cuda_runtime.h>

inline __device__ float4 operator+(const float4 &_a, const float4 &_b)
{
	return make_float4(_a.x + _b.x, _a.y + _b.y, _a.z + _b.z, _a.w + _b.w);
}

inline __device__ float4 operator-(const float4 &_a, const float4 &_b)
{
	return make_float4(_a.x - _b.x, _a.y - _b.y, _a.z - _b.z, _a.w - _b.w);
}

inline __device__ void operator+=(float4 &_a, const float4 &_b)
{
	_a.x += _b.x;
	_a.y += _b.y;
	_a.z += _b.z;
	_a.w += _b.w;
}

inline __device__ void operator*=(float4 &_a, const float4 &_b)
{
	_a.x *= _b.x;
	_a.y *= _b.y;
	_a.z *= _b.z;
	_a.w *= _b.w;
}

inline __device__ void operator*=(float4 &_a, const float &_b)
{
	_a.x *= _b;
	_a.y *= _b;
	_a.z *= _b;
	_a.w *= _b;
}

inline __device__ void operator/=(float4 &_a, const float &_b)
{
	_a.x /= _b;
	_a.y /= _b;
	_a.z /= _b;
	_a.w /= _b;
}

inline __device__ float4 operator/(const float &_a, float4 &_b)
{
	return make_float4(_a / _b.x,
										 _a / _b.y,
										 _a / _b.z,
										 _a / _b.w);
}

inline __device__ float4 operator*(const float4 &_a, const float4 &_b)
{
	return make_float4(_a.x*_b.x, _a.y*_b.y, _a.z*_b.z, _a.w);
}

inline __device__ float4 operator*(const float4 &_a, const float &_b)
{
	return make_float4(_a.x*_b, _a.y*_b, _a.z*_b, _a.w);
}

inline __device__ float4 operator*(const float &_a, const float4 &_b)
{
	return make_float4(_a*_b.x, _a*_b.y, _a*_b.z, _b.w);
}

inline __device__ float dot(const float4 &_a, const float4 &_b)
{
	return _a.x*_b.x + _a.y*_b.y + _a.z*_b.z;
}

inline __device__ float4 cross(const float4 &_a, const float4 &_b)
{
	return make_float4(_a.y * _b.z - _a.z * _b.y, _a.z * _b.x - _a.x * _b.z, _a.x * _b.y - _a.y * _b.x, 0.f);
}

inline __device__ float4 normalize(const float4 &_a)
{
	float invLen = rsqrtf(dot(_a, _a));
	return _a * invLen;
}

inline __device__ float clamp(const float &_val, const float &_low, const float &_hi)
{
	return max(_low, min(_hi, _val));
}

inline __device__ float4 clamp(const float4 &_val, const float &_low, const float &_hi)
{
	float x = clamp(_val.x, _low, _hi);
	float y = clamp(_val.y, _low, _hi);
	float z = clamp(_val.z, _low, _hi);
	return make_float4(x, y, z, _val.w);
}

inline __device__ float distanceSquared(const float4 &_v)
{
	return dot(_v, _v);
}

inline __device__ float distance(const float4 &_v)
{
	return sqrtf(dot(_v, _v));
}
