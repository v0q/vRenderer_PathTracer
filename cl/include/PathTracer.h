#pragma once

typedef struct Ray {
  float4 m_origin;
  float4 m_dir;
} Ray;

typedef struct Sphere {
  float m_r;
	float4 m_pos;
	float4 m_emission;
	float4 m_col;
} Sphere;

typedef struct vCamera {
  float4 m_origin;
  float4 m_dir;
  float4 m_upV;
  float4 m_rightV;
  float m_fovScale;
} vCamera;

typedef struct vHitData {
	float4 m_hitPoint;
	float4 m_normal;
	float4 m_emission;
	float4 m_color;
  float2 m_uv;
  unsigned int m_type;
} vHitData;

float intersectSphere(const Sphere *_sphere, const Ray *_ray);
float4 intersectTriangle(const float4 _v1, const float4 _v2, const float4 _v3, const Ray *_ray);
inline bool intersectNearAndFar(const float2 _ray, const float2 _limits, float *_tNear, float *_tFar);
bool intersectCFBVH(const Ray *_ray, const float3 _bottom, const float3 _top, float *_tNear, float *_tFar);

Ray createRay(float4 _o, float4 _d);
bool intersectScene(const Ray *_ray, __global const float4 *_vertices, __global const float4 *_normals, __global const float4 *_bvhNodes, vHitData *_hitData);
static float get_random(unsigned int *_seed0, unsigned int *_seed1);
//float4 trace(const Ray* _camray, __read_only image1d_t _vertices, __read_only image1d_t _normals, __read_only image1d_t _bvhNodes, __read_only image1d_t _triIdxList, unsigned int *_seed0, unsigned int *_seed1);
float4 trace(const Ray* _camray, __global const float4 *_vertices, __global const float4 *_normals, __global const float4 *_bvhNodes, __read_only image2d_t _hdr, unsigned int *_seed0, unsigned int *_seed1);

