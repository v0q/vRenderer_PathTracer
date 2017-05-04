///
/// \file MathHelpers.cuh
/// \brief Device math helper functions
/// \authors Teemu Lindborg
/// \version 1.0
/// \date 04/05/17 Updated to NCCA Coding standard
/// Revision History :
/// Initial Version 08/12/16
/// \todo -
///

#pragma once

#include <cuda_runtime.h>

__constant__ __device__ float PI = 3.14159265359f;
__constant__ __device__ float epsilon = 0.0000000003f;

///
/// \brief Simple mat4 struct used for normal map calculations
///
typedef struct mat4
{
	///
	/// \brief mat4 Default ctor for the matrix
	/// \param _a First row of the matrix
	/// \param _b Second row of the matrix
	/// \param _c Third row of the matrix
	/// \param _d Fourth row of the matrix
	/// \return
	///
	__device__ mat4(const float4 _a = make_float4(1.f, 0.f, 0.f, 0.f),
									const float4 _b = make_float4(0.f, 1.f, 0.f, 0.f),
									const float4 _c = make_float4(0.f, 0.f, 1.f, 0.f),
									const float4 _d = make_float4(0.f, 0.f, 0.f, 1.f)) :
		m_0(_a),
		m_1(_b),
		m_2(_c),
		m_3(_d)
	{}

	///
	/// \brief m_0 First row of the matrix
	///
	float4 m_0;

	///
	/// \brief m_1 Second row of the matrix
	///
	float4 m_1;

	///
	/// \brief m_2 Third row of the matrix
	///
	float4 m_2;

	///
	/// \brief m_3 Fourth row of the matrix
	///
	float4 m_3;
} mat4;

// Mat4
///
/// \brief operator* Matrix-Vector multiplication, used for normal map calculations
/// \param _a Matrix component
/// \param _b Vector component
/// \return Resulting vector
///
inline __device__ float4 operator*(const mat4 &_a, const float4 &_b)
{
	return make_float4(_a.m_0.x * _b.x + _a.m_1.x * _b.y + _a.m_2.x * _b.z + _a.m_3.x * _b.w,
										 _a.m_0.y * _b.x + _a.m_1.y * _b.y + _a.m_2.y * _b.z + _a.m_3.y * _b.w,
										 _a.m_0.z * _b.x + _a.m_1.z * _b.y + _a.m_2.z * _b.z + _a.m_3.z * _b.w,
										 _a.m_0.w * _b.x + _a.m_1.w * _b.y + _a.m_2.w * _b.z + _a.m_3.w * _b.w);
}

// Float 4
///
/// \brief operator+ Vector-Vector addition
/// \param _a First vector
/// \param _b second vector
/// \return Resulting vector
///
inline __device__ float4 operator+(const float4 &_a, const float4 &_b)
{
	return make_float4(_a.x + _b.x, _a.y + _b.y, _a.z + _b.z, _a.w + _b.w);
}

///
/// \brief operator- Vector-Vector subtraction
/// \param _a First vector
/// \param _b Second vector
/// \return Resulting vector
///
inline __device__ float4 operator-(const float4 &_a, const float4 &_b)
{
	return make_float4(_a.x - _b.x, _a.y - _b.y, _a.z - _b.z, _a.w - _b.w);
}

///
/// \brief operator+= Vector-Vector addition
/// \param _a First vector
/// \param _b Second vector
///
inline __device__ void operator+=(float4 &_a, const float4 &_b)
{
	_a.x += _b.x;
	_a.y += _b.y;
	_a.z += _b.z;
	_a.w += _b.w;
}

///
/// \brief operator*= Vector-Vector multiplication
/// \param _a First vector
/// \param _b Second vector
///
inline __device__ void operator*=(float4 &_a, const float4 &_b)
{
	_a.x *= _b.x;
	_a.y *= _b.y;
	_a.z *= _b.z;
	_a.w *= _b.w;
}

///
/// \brief operator+= Vector-Float addition
/// \param _a First vector
/// \param _b Float to add to vector components
///
inline __device__ void operator*=(float4 &_a, const float &_b)
{
	_a.x *= _b;
	_a.y *= _b;
	_a.z *= _b;
}

///
/// \brief operator/= Vector-Float division
/// \param _a First vector
/// \param _b Divisor float
///
inline __device__ void operator/=(float4 &_a, const float &_b)
{
	_a.x /= _b;
	_a.y /= _b;
	_a.z /= _b;
}

///
/// \brief operator/ Float-Vector division
/// \param _a Float
/// \param _b Vector divisor
/// \return Resulting vector
///
inline __device__ float4 operator/(const float &_a, float4 &_b)
{
	return make_float4(_a / _b.x,
										 _a / _b.y,
										 _a / _b.z,
										 _a / _b.w);
}

///
/// \brief operator* Vector-Vector multiplication
/// \param _a First vector
/// \param _b Second vector
/// \return Resulting vector
///
inline __device__ float4 operator*(const float4 &_a, const float4 &_b)
{
	return make_float4(_a.x*_b.x, _a.y*_b.y, _a.z*_b.z, _a.w);
}

///
/// \brief operator* Vector-Float multiplication
/// \param _a First vector
/// \param _b Float multiplier
/// \return Resulting vector
///
inline __device__ float4 operator*(const float4 &_a, const float &_b)
{
	return make_float4(_a.x*_b, _a.y*_b, _a.z*_b, _a.w);
}

///
/// \brief operator* Float-Vector multiplication
/// \param _a First vector
/// \param _b Vector multiplier
/// \return Resulting vector
///
inline __device__ float4 operator*(const float &_a, const float4 &_b)
{
	return make_float4(_a*_b.x, _a*_b.y, _a*_b.z, _b.w);
}

// Float2
///
/// \brief operator+ Vector-Vector addition
/// \param _a First vector
/// \param _b Second vector
/// \return Resulting vector
///
inline __device__ float2 operator+(const float2 &_a, const float2 &_b)
{
	return make_float2(_a.x + _b.x, _a.y + _b.y);
}

///
/// \brief operator- Vector-Vector subtraction
/// \param _a First vector
/// \param _b Second vector
/// \return Resulting vector
///
inline __device__ float2 operator-(const float2 &_a, const float2 &_b)
{
	return make_float2(_a.x - _b.x, _a.y - _b.y);
}

///
/// \brief operator+= Vector-Vector addition
/// \param _a First vector
/// \param _b Second vector
///
inline __device__ void operator+=(float2 &_a, const float2 &_b)
{
	_a.x += _b.x;
	_a.y += _b.y;
}

///
/// \brief operator*= Vector-Vector multiplication
/// \param _a First vector
/// \param _b Second vector
///
inline __device__ void operator*=(float2 &_a, const float2 &_b)
{
	_a.x *= _b.x;
	_a.y *= _b.y;
}

///
/// \brief operator*= Vector-Float multiplication
/// \param _a First vector
/// \param _b Float multiplier
///
inline __device__ void operator*=(float2 &_a, const float &_b)
{
	_a.x *= _b;
	_a.y *= _b;
}

///
/// \brief operator/= Vector-Vector divison
/// \param _a First vector
/// \param _b Vector divisor
///
inline __device__ void operator/=(float2 &_a, const float &_b)
{
	_a.x /= _b;
	_a.y /= _b;
}

///
/// \brief operator/ Float-Vector divison
/// \param _a Float
/// \param _b Vector divisor
/// \return Resulting vector
///
inline __device__ float2 operator/(const float &_a, float2 &_b)
{
	return make_float2(_a / _b.x,
										 _a / _b.y);
}

///
/// \brief operator* Vector-Vector multiplication
/// \param _a First vector
/// \param _b Second vector
/// \return Resulting vector
///
inline __device__ float2 operator*(const float2 &_a, const float2 &_b)
{
	return make_float2(_a.x*_b.x, _a.y*_b.y);
}

///
/// \brief operator* Vector-Float multiplication
/// \param _a First vector
/// \param _b Float multiplier
/// \return Resulting vector
///
inline __device__ float2 operator*(const float2 &_a, const float &_b)
{
	return make_float2(_a.x*_b, _a.y*_b);
}

///
/// \brief operator* Float-Vector multiplication
/// \param _a Float
/// \param _b Vector multiplier
/// \return Resulting vector
///
inline __device__ float2 operator*(const float &_a, const float2 &_b)
{
	return make_float2(_a*_b.x, _a*_b.y);
}

///
/// \brief operator/= Vector-Vector division
/// \param _a First vector
/// \param _b Second vector
/// \return Resulting vector
///
inline __device__ void operator/=(float2 &_a, const float2 &_b)
{
	_a.x /= _b.x;
	_a.y /= _b.y;
}

///
/// \brief dot Dot product
/// \param _a First vector
/// \param _b Second vector
/// \return Result of the dot product
///
inline __device__ float dot(const float4 &_a, const float4 &_b)
{
	return _a.x*_b.x + _a.y*_b.y + _a.z*_b.z;
}

///
/// \brief cross Vector cross product
/// \param _a First vector
/// \param _b Second vector
/// \return Resulting vector
///
inline __device__ float4 cross(const float4 &_a, const float4 &_b)
{
	return make_float4(_a.y * _b.z - _a.z * _b.y, _a.z * _b.x - _a.x * _b.z, _a.x * _b.y - _a.y * _b.x, 0.f);
}

///
/// \brief normalize Normalizes a given vector
/// \param _a Vector to normalize
/// \return Vector with the same direction as _a but length of 1
///
inline __device__ float4 normalize(const float4 &_a)
{
	float invLen = rsqrtf(dot(_a, _a));
	return _a * invLen;
}

///
/// \brief clamp Clamps an integer between two values
/// \param _val Value to clamp
/// \param _low Lower limit of the clamp
/// \param _hi Higher limit of the clamp
/// \return Clamped value
///
inline __device__ float clamp(const int &_val, const int &_low, const int &_hi)
{
	return max(_low, min(_hi, _val));
}


/// \brief clamp Clamps a float between two values
/// \param _val Value to clamp
/// \param _low Lower limit of the clamp
/// \param _hi Higher limit of the clamp
/// \return Clamped value
///
inline __device__ float clamp(const float &_val, const float &_low, const float &_hi)
{
	return max(_low, min(_hi, _val));
}

///
/// \brief clamp Clamps a vector between two values
/// \param _val Vector to clamp
/// \param _low Lower limit of the clamp
/// \param _hi Higher limit of the clamp
/// \return Clamped vector
///
inline __device__ float4 clamp(const float4 &_val, const float &_low, const float &_hi)
{
	float x = clamp(_val.x, _low, _hi);
	float y = clamp(_val.y, _low, _hi);
	float z = clamp(_val.z, _low, _hi);
	return make_float4(x, y, z, _val.w);
}

///
/// \brief max Returns a component-wise max vector from two vectors
/// \param _a First vector
/// \param _b Second vector
/// \return Vector with maximum components from the two given vectors
///
inline __device__ float4 max(const float4 &_a, const float4 &_b)
{
	return make_float4(max(_a.x, _b.x), max(_a.y, _b.y), max(_a.z, _b.z), max(_a.w, _b.w));
}

///
/// \brief distanceSquared Gets the squared distance of a given vector
/// \param _v Vector to get the squared distance from
/// \return Squared length
///
inline __device__ float distanceSquared(const float4 &_v)
{
	return dot(_v, _v);
}

///
/// \brief distance Calculates and returns the length of a given vector
/// \param _v Vector to check
/// \return Length of the vector
///
inline __device__ float distance(const float4 &_v)
{
	return sqrtf(dot(_v, _v));
}

///
/// \brief powf Calculate a vector to a given power
/// \param _a Vector whose components to raise
/// \param _pow Power to raise the vector's components to
/// \return Resulting vector
///
inline __device__ float4 powf(const float4 &_a, const float &_pow)
{
	return make_float4(powf(_a.x, _pow),
										 powf(_a.y, _pow),
										 powf(_a.z, _pow),
										 powf(_a.w, _pow));
}

///
/// \brief lerp Calculate the linear interpolation between the two given values
/// \param _a Value a
/// \param _b Value b
/// \param _w Alpha
/// \return Interpolated value
///
inline __device__ float lerp(const float &_a, const float &_b, const float &_w)
{
	return (1.f - _w) * _a + _w * _b;
}

///
/// Based on the CudaTracerLib by Hannes Hergeth (https://github.com/hhergeth/CudaTracerLib)
///
inline __device__ int min_min(const int &_a, const int &_b, const int &_c)
{
	int v;
	asm(
		"vmin.s32.s32.s32.min %0, %1, %2, %3;"
	:
		"=r"(v) :
			"r"(_a),
			"r"(_b),
			"r"(_c));

	return v;
}

inline __device__ int min_max(const int &_a, const int &_b, const int &_c)
{
	int v;
	asm(
		"vmin.s32.s32.s32.max %0, %1, %2, %3;"
	:
		"=r"(v) :
			"r"(_a),
			"r"(_b),
			"r"(_c));

	return v;
}

inline __device__ int max_min(const int &_a, const int &_b, const int &_c)
{
	int v;
	asm("vmax.s32.s32.s32.min %0, %1, %2, %3;"
	:
		"=r"(v) :
			"r"(_a),
			"r"(_b),
			"r"(_c));

	return v;
}

inline __device__ int max_max(const int &_a, const int &_b, const int &_c)
{
	int v;
	asm("vmax.s32.s32.s32.max %0, %1, %2, %3;"
	:
		"=r"(v) :
			"r"(_a),
			"r"(_b),
			"r"(_c));

	return v;
}

inline __device__ float fmin_fmin(const float &_a, const float &_b, const float &_c)
{
	return __int_as_float(
												min_min(__float_as_int(_a),
																__float_as_int(_b),
																__float_as_int(_c))
											);
}

inline __device__ float fmin_fmax(const float &_a, const float &_b, const float &_c)
{
	return __int_as_float(
												min_max(__float_as_int(_a),
																__float_as_int(_b),
																__float_as_int(_c))
											);
}

inline __device__ float fmax_fmin(const float &_a, const float &_b, const float &_c)
{
	return __int_as_float(
												max_min(__float_as_int(_a),
																__float_as_int(_b),
																__float_as_int(_c))
											);
}

inline __device__ float fmax_fmax(const float &_a, const float &_b, const float &_c)
{
	return __int_as_float(
												max_max(__float_as_int(_a),
																__float_as_int(_b),
																__float_as_int(_c))
											);
}

inline __device__ float spanBeginKepler(const float &_a0, const float &_a1, const float &_b0, const float &_b1, const float &_c0, const float &_c1, const float &_d)
{
	return fmax_fmax(min(_a0, _a1), min(_b0, _b1), fmin_fmax(_c0, _c1, _d));
}

inline __device__ float spanEndKepler(const float &_a0, const float &_a1, const float &_b0, const float &_b1, const float &_c0, const float &_c1, const float &_d)
{
	return fmin_fmin(max(_a0, _a1), max(_b0, _b1), fmax_fmin(_c0, _c1, _d));
}
