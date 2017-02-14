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

///
/// \brief intersectBoundingBox Based on the Ray-Box Intersection chapter from http://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
/// \param _ray
/// \return
///
__device__ bool intersectBVH(const vBVH _bb, const Ray *_ray)
{
	float tNear = -FLT_MAX;
	float tFar = FLT_MAX;

#define SWAP(x, y)	\
	{									\
		float tmp = x;	\
		x = y;					\
		y = tmp;				\
	}

	for(uint8_t i = 0; i < kNumPlaneSetNormals; ++i)
	{
		float numerator = dot(_bb.m_normal[i], _ray->m_origin);
		float denominator = dot(_bb.m_normal[i], _ray->m_dir);

		float tn = (_bb.m_dNear[i] - numerator) / denominator;
		float tf = (_bb.m_dFar[i] - numerator) / denominator;
		if(denominator < 0)
			SWAP(tn, tf);
		if(tn > tNear)
			tNear = tn;
		if(tf < tFar)
			tFar = tf;
		if(tNear > tFar)
			return false;
	}



//	tx.x = (_bb.m_x.x - _ray->m_origin.x) / _ray->m_dir.x;
//	tx.y = (_bb.m_x.y - _ray->m_origin.x) / _ray->m_dir.x;
//	if(tx.x > tx.y) SWAP(tx.x, tx.y)

//	ty.x = (_bb.m_y.x - _ray->m_origin.y) / _ray->m_dir.y;
//	ty.y = (_bb.m_y.y - _ray->m_origin.y) / _ray->m_dir.y;
//	if(ty.x > ty.y) SWAP(ty.x, ty.y)

//	if ((tx.x > ty.y) || (ty.x > tx.y))
//		return false;
//	if (ty.x > tx.x)
//		tx.x = ty.x;
//	if (ty.y < tx.y)
//		tx.y = ty.y;

//	tz.x = (_bb.m_z.x - _ray->m_origin.z) / _ray->m_dir.z;
//	tz.y = (_bb.m_z.y - _ray->m_origin.z) / _ray->m_dir.z;
//	if(tz.x > tz.y) SWAP(tz.x, tz.y)

//	if ((tx.x > tz.y) || (tz.x > tx.y))
//		return false;
//	if (tz.x > tx.x)
//		tx.x = tz.x;
//	if (tz.y < tx.y)
//		tx.y = tz.y;

//	if(tx.x < 0 && tx.y < 0)
//		return false;



//	for (uint8_t i = 0; i < kNumPlaneSetNormals; ++i) {
//		float tn = (d[i][0] - precomputedNumerator[i]) / precomputeDenominator[i];
//		float tf = (d[i][1] - precomputedNumerator[i]) / precomputeDenominator[i];
//		if (precomputeDenominator[i] < 0) std::swap(tn, tf);
//		if (tn > tNear) tNear = tn, planeIndex = i;
//		if (tf < tFar) tFar = tf;
//		if (tNear > tFar) return false;
//	}

	return true;
}
