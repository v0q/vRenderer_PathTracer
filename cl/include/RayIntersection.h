#pragma once

#include "cl/include/PathTracer.h"

__constant float PI = 3.14159265359f;
__constant float epsilon = 0.0003f;

float intersectSphere(const Sphere *_sphere, const Ray *_ray)
{
  float4 rayToCenter = _sphere->m_pos - _ray->m_origin;
  float b = dot(rayToCenter, _ray->m_dir);
  float c = dot(rayToCenter, rayToCenter) - _sphere->m_r*_sphere->m_r;
  float disc = b * b - c;

  if (disc < 0.0f) return 0.0f;
  else disc = sqrt(disc);

  if ((b - disc) > epsilon) return b - disc;
  if ((b + disc) > epsilon) return b + disc;

  return 0.0f;
}

///
/// \brief intersect
/// \param _ray One axis direction component stored in x and origin in y
/// \param _limits One axis bottom component stored in x and top in y
/// \return False if no intersection was found and we can exit, true if we need to continue
///
inline bool intersectNearAndFar(const float2 _ray, const float2 _limits, float *_tNear, float *_tFar)
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
    if(t1 > *_tNear)
      *_tNear = t1;
    if(t2 < *_tFar)
      *_tFar = t2;
    if(*_tNear > *_tFar)
      return false;
    if(*_tFar < 0.f)
      return false;
  }

  return true;
}

bool intersectCFBVH(const Ray *_ray, const float3 _bottom, const float3 _top, float *_tNear, float *_tFar)
{
  *_tNear = -FLT_MAX;
  *_tFar = FLT_MAX;

  // X
  if(!intersectNearAndFar((float2)(_ray->m_dir.x, _ray->m_origin.x),
                          (float2)(_bottom.x, _top.x), _tNear, _tFar))
    return false;

  // Y
  if(!intersectNearAndFar((float2)(_ray->m_dir.y, _ray->m_origin.y),
                          (float2)(_bottom.y, _top.y), _tNear, _tFar))
    return false;

  // Z
  if(!intersectNearAndFar((float2)(_ray->m_dir.z, _ray->m_origin.z),
                          (float2)(_bottom.z, _top.z), _tNear, _tFar))
    return false;

  return true;
}
