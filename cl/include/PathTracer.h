#pragma once

typedef enum Refl_t { SPEC, DIFF } Refl_t;
typedef enum vTextureType { DIFFUSE, NORMAL, SPECULAR } vTextureType;

typedef struct Ray {
  float4 m_origin;
  float4 m_dir;
} Ray;

typedef struct Sphere {
  float m_r;
	float4 m_pos;
	float4 m_emission;
	float4 m_col;
  Refl_t m_refl;
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
  float4 m_specularColor;
  unsigned int m_hitType;
} vHitData;

unsigned int floatAsInt(const float _a);
Ray createRay(float4 _o, float4 _d);
static float get_random(unsigned int *_seed0, unsigned int *_seed1);

bool intersectScene(const Ray *_ray, __global const float4 *_vertices, __global const float4 *_normals, __global const float4 *_tangents, __global const float4 *_bvhNodes, __global const float2 *_uvs, __read_only image2d_t _diffuse, __read_only image2d_t _normal, __read_only image2d_t _specular,  bool _hasDiffuseMap, bool _hasNormalMap, bool _hasSpecularMap, vHitData *_hitData);
float4 trace(const Ray* _camray, __global const float4 *_vertices, __global const float4 *_normals, __global const float4 *_tangents, __global const float4 *_bvhNodes, __global const float2 *_uvs, __read_only image2d_t _hdr, __read_only image2d_t _diffuse, __read_only image2d_t _normal, __read_only image2d_t _specular,  bool _hasDiffuseMap, bool _hasNormalMap, bool _hasSpecularMap, unsigned int *_seed0, unsigned int *_seed1);
