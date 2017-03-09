#pragma once

#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "PathTracer.cuh"
#include "MathHelpers.cuh"

#define kNumPlaneSetNormals 7

__constant__ __device__ float PI = 3.14159265359f;
__constant__ __device__ float epsilon = 0.0000000003f;

typedef struct Ray {
	float4 m_origin;
	float4 m_dir;

	__device__ Ray(float4 _o, float4 _d) : m_origin(_o), m_dir(_d) {}
} Ray;

__device__ float intersectTriangle(const float4 &_v1, const float4 &_v2, const float4 &_v3, const Ray *_ray)
{
	float4 e1, e2;  //Edge1, Edge2
	float4 p, q, t;
	float det, inv_det, u, v;
	float dist;

	//Find vectors for two edges sharing V1
	e1 = _v2 - _v1;
	e2 = _v3 - _v1;
	//Begin calculating determinant - also used to calculate u parameter

	p = cross(_ray->m_dir, e2);
	//if determinant is near zero, ray lies in plane of triangle or ray is parallel to plane of triangle
	det = dot(e1, p);
	//NOT CULLING
	if(det > -epsilon && det < epsilon)
		return 0.f;
	inv_det = 1.f / det;

	//calculate distance from V1 to ray origin
	t = _ray->m_origin - _v1;

	//Calculate u parameter and test bound
	u = dot(t, p) * inv_det;
	//The intersection lies outside of the triangle
	if(u < 0.f || u > 1.f)
		return 0.f;

	//Prepare to test v parameter
	q = cross(t, e1);

	//Calculate V parameter and test bound
	v = dot(_ray->m_dir, q) * inv_det;
	//The intersection lies outside of the triangle
	if(v < 0.f || u + v  > 1.f)
		return 0.f;

	dist = dot(e2, q) * inv_det;

	if(dist > epsilon)
		return dist;

	// No hit, no win
	return 0.f;
}

///
/// \brief intersect
/// \param _ray One axis direction component stored in x and origin in y
/// \param _limits One axis bottom component stored in x and top in y
/// \return False if no intersection was found and we can exit, true if we need to continue
///
inline __device__ bool intersectNearAndFar(const float2 &_ray, const float2 &_limits, float &_tNear, float &_tFar)
{
	// box intersection routine
	if(_ray.x == 0.f)
	{
		if(_ray.y < _limits.x)
			return false;					    \
		if(_ray.y > _limits.y)
			return false;
	}
	else
	{
		float t1 = (_limits.x - _ray.y) / _ray.x;
		float t2 = (_limits.y - _ray.y) / _ray.x;
		if(t1 > t2)
		{
			float tmp = t1;
			t1 = t2;
			t2 = tmp;
		}
		if(t1 > _tNear)
			_tNear = t1;
		if(t2 < _tFar)
			_tFar = t2;
		if(_tNear > _tFar)
			return false;
		if(_tFar < 0.f)
			return false;
	}

	return true;
}

__device__ bool intersectCFBVH(const Ray *_ray, const float3 &_bottom, const float3 &_top)
{
	float Tnear = -FLT_MAX;
	float Tfar = FLT_MAX;

	// X
	if(!intersectNearAndFar(make_float2(_ray->m_dir.x, _ray->m_origin.x),
													make_float2(_bottom.x, _top.x), Tnear, Tfar))
		return false;

	// Y
	if(!intersectNearAndFar(make_float2(_ray->m_dir.y, _ray->m_origin.y),
													make_float2(_bottom.y, _top.y), Tnear, Tfar))
		return false;

	// Z
	if(!intersectNearAndFar(make_float2(_ray->m_dir.z, _ray->m_origin.z),
													make_float2(_bottom.z, _top.z), Tnear, Tfar))
		return false;

	return true;
}
