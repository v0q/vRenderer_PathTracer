///
/// \file RayIntersection.cuh
/// \brief Ray-XX intersection functions
/// \authors Teemu Lindborg
/// \version 1.0
/// \date 04/05/17 Updated to NCCA Coding standard
/// Revision History :
/// Initial Version 08/12/16
/// \todo -
///

#pragma once

#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "PathTracer.cuh"
#include "MathHelpers.cuh"

///
/// \brief Ray Simple structure for a ray
///
typedef struct Ray
{
	///
	/// \brief m_origin Origin of the current ray
	///
	float4 m_origin;

	///
	/// \brief m_dir Direction of the ray
	///
	float4 m_dir;

	///
	/// \brief Ray Default ctor for a ray
	/// \param _o Origin of the ray
	/// \param _d Direction of the ray
	///
	__device__ Ray(float4 _o, float4 _d) : m_origin(_o), m_dir(_d) {}
} Ray;

///
/// \brief intersectTriangle Calculate the ray-triangle intersection and barycentric coordinates for texturing.
///													 Based on the "Fast, Minimum Storage Ray/Triangle Intersection" by Tomas Akenine-Möller et al.
///													 Available at: http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/raytri_tam.pdf,
///													 Supplemental source code available at: http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/raytri/raytri.c
/// \param _v1 Vertex 1 of the triangle
/// \param _v2 Vertex 2 of the triangle
/// \param _v3 Vertex 3 of the triangle
/// \param _ray Ray to test the triangle against
/// \return Distance to the triangle and barycentric coordinates if an intersection was found
///
__device__ float4 intersectTriangle(const float4 &_v1, const float4 &_v2, const float4 &_v3, const Ray *_ray)
{
	float4 e1, e2;  //Edge1, Edge2
	float4 p, q, t;
	float det, inv_det, u, v;
	float dist;

	// Find vectors for two edges sharing V1
	e1 = _v2 - _v1;
	e2 = _v3 - _v1;

	// Begin calculating determinant - also used to calculate u parameter
	p = cross(_ray->m_dir, e2);
	// If determinant is near zero, ray lies in plane of triangle or ray is parallel to plane of triangle
	det = dot(e1, p);
	// NOT CULLING
	if(det > -epsilon && det < epsilon)
	{
		return make_float4(0.f, 0.f, 0.f, 0.f);
	}

	inv_det = 1.f / det;

	// Calculate distance from V1 to ray origin
	t = _ray->m_origin - _v1;

	// Calculate u parameter and test bound
	u = dot(t, p) * inv_det;

	// The intersection lies outside of the triangle
	if(u < 0.f || u > 1.f)
	{
		return make_float4(0.f, 0.f, 0.f, 0.f);
	}

	// Prepare to test v parameter
	q = cross(t, e1);

	// Calculate V parameter and test bound
	v = dot(_ray->m_dir, q) * inv_det;

	// The intersection lies outside of the triangle
	if(v < 0.f || u + v  > 1.f)
	{
		return make_float4(0.f, 0.f, 0.f, 0.f);
	}

	dist = dot(e2, q) * inv_det;

	// Intersection was found, return the distance and barycentric coordinates
	if(dist > epsilon)
	{
		return make_float4(dist, u, v, 0.f);
	}

	// No hit, no win
	return make_float4(0.f, 0.f, 0.f, 0.f);
}
