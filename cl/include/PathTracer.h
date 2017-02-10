#pragma once

typedef struct Ray {
  float4 m_origin;
  float4 m_dir;
  float4 m_invDir;
  uint3 m_sign;
} Ray;

typedef struct Sphere {
  float m_r;
	float4 m_pos;
	float4 m_emission;
	float4 m_col;
} Sphere;

typedef struct vBoundingBox {
  float2 m_x;
  float2 m_y;
  float2 m_z;
} vBoundingBox;

typedef struct vVert {
	float4 m_vert;
	float4 m_normal;
} vVert;

typedef struct vTriangle {
  vVert m_v1;
  vVert m_v2;
  vVert m_v3;
} vTriangle;

typedef struct vMesh {
  __global vTriangle *m_mesh;
  vBoundingBox m_bb;
  unsigned int m_triCount;
} vMesh;

typedef struct vHitData {
	float4 m_hitPoint;
	float4 m_normal;
	float4 m_emission;
	float4 m_color;
} vHitData;

Ray createRay(float4 _o, float4 _d);
bool intersectScene(const Ray *_ray, __global const vMesh *_scene, vHitData *_hitData);
float intersectTriangle(const float4 _v1, const float4 _v2, const float4 _v3, const Ray *_ray);
float intersectSphere(const Sphere *_sphere, const Ray *_ray);
bool intersectBoundingBox(const Ray *_ray, const float2 _x, const float2 _y, const float2 _z);
static float get_random(unsigned int *_seed0, unsigned int *_seed1);
float4 trace(const Ray* _camray, __global const vMesh *_scene, unsigned int *_seed0, unsigned int *_seed1);

