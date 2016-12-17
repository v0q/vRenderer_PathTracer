#pragma once

struct Ray {
  float3 m_origin;
  float3 m_dir;
};

enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()

struct Sphere {
  float m_r;       // radius
  float3 m_pos;
  float3 m_emission;
  float3 m_col;
  enum Refl_t m_refl;
};

struct Ray createRay(float3 _o, float3 _d);
float intersectSphere(const struct Sphere _sphere, const struct Ray _r);
float3 radiance(struct Ray *_r, unsigned int *s0, unsigned int *s1);
