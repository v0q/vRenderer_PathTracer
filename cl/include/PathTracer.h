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

Ray createRay(float3 _o, float3 _d);
float intersect_sphere(const Sphere *_sphere, const Ray *_ray);
bool intersect_scene(const Ray *_ray, float *_t, int *_id);
static float get_random(unsigned int *_seed0, unsigned int *_seed1);
float3 trace(const Ray* _camray, unsigned int *_seed0, unsigned int *_seed1);
//float intersectSphere(const struct Sphere _sphere, const struct Ray _r);
//float3 radiance(struct Ray *_r, unsigned int *s0, unsigned int *s1);
