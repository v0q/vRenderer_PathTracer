#pragma once

#include "cl/include/PathTracer.h"

__constant float PI = 3.14159265359f;
__constant float EPSILON = 0.0000003f;

//float intersectTriangle(const float4 _v1, const float4 _v2, const float4 _v3, const Ray *_ray);
//float intersectSphere(const Sphere *_sphere, const Ray *_ray);

float intersectTriangle(const float4 _v1, const float4 _v2, const float4 _v3, const Ray *_ray)
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

float intersectSphere(const Sphere *_sphere, const Ray *_ray)
{
  float4 rayToCenter = _sphere->m_pos - _ray->m_origin;
  float b = dot(rayToCenter, _ray->m_dir);
  float c = dot(rayToCenter, rayToCenter) - _sphere->m_r*_sphere->m_r;
  float disc = b * b - c;

  if (disc < 0.0f) return 0.0f;
  else disc = sqrt(disc);

  if ((b - disc) > EPSILON) return b - disc;
  if ((b + disc) > EPSILON) return b + disc;

  return 0.0f;
}

bool intersectBoundingBox(const Ray *_ray, const float2 _x, const float2 _y, const float2 _z)
{
  float2 tx;
  float2 ty;
  float2 tz;

  tx.x = ((_ray->m_sign.x ? _x.y : _x.x) - _ray->m_origin.x) * _ray->m_invDir.x;
  tx.y = ((_ray->m_sign.x ? _x.x : _x.y) - _ray->m_origin.x) * _ray->m_invDir.x;
  ty.x = ((_ray->m_sign.y ? _y.y : _y.x) - _ray->m_origin.y) * _ray->m_invDir.y;
  ty.y = ((_ray->m_sign.y ? _y.x : _y.y) - _ray->m_origin.y) * _ray->m_invDir.y;

  if ((tx.x > ty.y) || (ty.x > tx.y))
    return false;
  if (ty.x > tx.x)
    tx.x = ty.x;
  if (ty.y < tx.y)
    tx.y = ty.y;

  tz.x = ((_ray->m_sign.z ? _z.y : _z.x) - _ray->m_origin.z) * _ray->m_invDir.z;
  tz.y = ((_ray->m_sign.z ? _z.x : _z.y) - _ray->m_origin.z) * _ray->m_invDir.z;

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
