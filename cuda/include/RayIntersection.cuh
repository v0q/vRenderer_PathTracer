#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "PathTracer.cuh"
#include "MathHelpers.cuh"

#define PI 3.14159265359f
#define EPSILON 0.0000003f

typedef struct Ray {
	float4 m_origin;
	float4 m_dir;
	float4 m_invDir;
	uint3 m_sign;

	__device__ Ray(float4 _o, float4 _d) : m_origin(_o), m_dir(_d) {
		m_invDir = 1.f / m_dir;
		m_sign.x = (m_invDir.x < 0);
		m_sign.y = (m_invDir.y < 0);
		m_sign.z = (m_invDir.z < 0);
	}
} Ray;

typedef struct Sphere {
	float m_r;       // radius
	float4 m_pos;
	float4 m_emission;
	float4 m_col;

	__device__ float intersect(const Ray *_r) const
	{ // returns distance, 0 if nohit
		float4 op = m_pos - _r->m_origin; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
		float t;
		float eps = 1e-4;
		float b = dot(op, _r->m_dir);
		float det = b*b - dot(op, op) + m_r*m_r;
		if(det < 0)
			return 0;
		else
			det = sqrtf(det);
		return (t = b-det) > eps ? t : ((t = b+det) > eps ? t : 0.0);
	}
} Sphere;

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
	if(det > -EPSILON && det < EPSILON)
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

	if(dist > EPSILON)
		return dist;

	// No hit, no win
	return 0.f;
}

///
/// \brief intersectBoundingBox Based on the Ray-Box Intersection chapter from http://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
/// \param _ray
/// \return
///
__device__ bool intersectBoundingBox(const vBoundingBox _bb, const Ray *_ray)
{
		float2 tx;
		float2 ty;
		float2 tz;

		tx.x = ((_ray->m_sign.x ? _bb.m_x.y : _bb.m_x.x) - _ray->m_origin.x) * _ray->m_invDir.x;
		tx.y = ((_ray->m_sign.x ? _bb.m_x.x : _bb.m_x.y) - _ray->m_origin.x) * _ray->m_invDir.x;
		ty.x = ((_ray->m_sign.y ? _bb.m_y.y : _bb.m_y.x) - _ray->m_origin.y) * _ray->m_invDir.y;
		ty.y = ((_ray->m_sign.y ? _bb.m_y.x : _bb.m_y.y) - _ray->m_origin.y) * _ray->m_invDir.y;

		if ((tx.x > ty.y) || (ty.x > tx.y))
			return false;
		if (ty.x > tx.x)
			tx.x = ty.x;
		if (ty.y < tx.y)
			tx.y = ty.y;

		tz.x = ((_ray->m_sign.z ? _bb.m_z.y : _bb.m_z.x) - _ray->m_origin.z) * _ray->m_invDir.z;
		tz.y = ((_ray->m_sign.z ? _bb.m_z.x : _bb.m_z.y) - _ray->m_origin.z) * _ray->m_invDir.z;

		if ((tx.x > tz.y) || (tz.x > tx.y))
			return false;
		if (tz.x > tx.x)
			tx.x = tz.x;
		if (tz.y < tx.y)
			tx.y = tz.y;

		if(tx.x < 0 && tx.y < 0)
			return false;

		return true;
}
