#pragma once

typedef struct Ray {
  float3 m_origin;
  float3 m_dir;
} Ray;

//enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()

typedef struct Sphere {
  float m_r;       // radius
  float3 m_pos;
  float3 m_emission;
  float3 m_col;
} Sphere;

typedef struct vVert {
  float3 m_vert;
  float3 m_normal;
} vVert;

typedef struct vTriangle {
  vVert m_v1;
  vVert m_v2;
  vVert m_v3;
} vTriangle;

typedef struct vHitData {
  float3 m_hitPoint;
  float3 m_normal;
  float3 m_emission;
  float3 m_color;
} vHitData;

Ray createRay(float3 _o, float3 _d);
float intersectTriangle(const float3 _v1, const float3 _v2, const float3 _v3, const Ray *_ray);
float intersectSphere(const Sphere *_sphere, const Ray *_ray);
bool intersectScene(const Ray *_ray, __global const vTriangle *_scene, unsigned int _triCount, vHitData *_hitData);
static float get_random(unsigned int *_seed0, unsigned int *_seed1);
float3 trace(const Ray* _camray, __global const vTriangle *_scene, unsigned int _triCount, unsigned int *_seed0, unsigned int *_seed1);
//float intersectSphere(const struct Sphere _sphere, const struct Ray _r);
//float3 radiance(struct Ray *_r, unsigned int *s0, unsigned int *s1);
