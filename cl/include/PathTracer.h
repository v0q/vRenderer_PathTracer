#pragma once

typedef struct Ray {
	float4 m_origin;
	float4 m_dir;
} Ray;

//enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()

typedef struct Sphere {
  float m_r;       // radius
	float4 m_pos;
	float4 m_emission;
	float4 m_col;
} Sphere;

typedef struct vVert {
	float4 m_vert;
	float4 m_normal;
} vVert;

typedef struct vTriangle {
  vVert m_v1;
  vVert m_v2;
  vVert m_v3;
} vTriangle;

typedef struct vHitData {
	float4 m_hitPoint;
	float4 m_normal;
	float4 m_emission;
	float4 m_color;
} vHitData;

Ray createRay(float4 _o, float4 _d);
float intersectTriangle(const float4 _v1, const float4 _v2, const float4 _v3, const Ray *_ray);
float intersectSphere(const Sphere *_sphere, const Ray *_ray);
bool intersectScene(const Ray *_ray, __global const vTriangle *_scene, unsigned int _triCount, vHitData *_hitData);
static float get_random(unsigned int *_seed0, unsigned int *_seed1);
float4 trace(const Ray* _camray, __global const vTriangle *_scene, unsigned int _triCount, unsigned int *_seed0, unsigned int *_seed1);

