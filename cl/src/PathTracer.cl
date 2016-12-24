#include "cl/include/PathTracer.h"

__constant float invGamma = 1.f/2.2f;
__constant float PI = 3.14159265359f;
__constant float EPSILON = 0.0000003f;
__constant Sphere spheres[] = {			//Scene: radius, position, emission, color, material
  { 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.075f, 0.f, 0.f }, { 0.75f, 0.0f, 0.0f } }, //Left
  { 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.f, 0.075f, 0.f }, { 0.0f, 0.75f, 0.0f } }, //Right
  { 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f } }, //Back
  { 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f } }, //Frnt
  { 1e5f, { 50.0f, 1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f } }, //Botm
  { 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f } }, //Top
  { 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f } }, // small sphere 1
  { 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f } }, // small sphere 2
  { 600.0f, { 50.0f, 681.6f - .77f, 81.6f }, { 2.0f, 1.8f, 1.6f }, { 0.0f, 0.0f, 0.0f } }  // Light
};

Ray createRay(float3 _o, float3 _d)
{
  Ray ray;
  ray.m_origin = _o;
  ray.m_dir = _d;
  return ray;
}

float intersect_sphere(const Sphere *_sphere, const Ray *_ray) /* version using local copy of sphere */
{
  float3 rayToCenter = _sphere->m_pos - _ray->m_origin;
  float b = dot(rayToCenter, _ray->m_dir);
  float c = dot(rayToCenter, rayToCenter) - _sphere->m_r*_sphere->m_r;
  float disc = b * b - c;

  if (disc < 0.0f) return 0.0f;
  else disc = sqrt(disc);

  if ((b - disc) > EPSILON) return b - disc;
  if ((b + disc) > EPSILON) return b + disc;

  return 0.0f;
}

bool intersect_scene(const Ray *_ray, float *_t, int *_id)
{
  /* initialise t to a very large number,
  so t will be guaranteed to be smaller
  when a hit with the scene occurs */

  int n = sizeof(spheres)/sizeof(Sphere);;
  float inf = 1e20f;
  *_t = inf;

  /* check if the ray intersects each sphere in the scene */
  for(int i = 0; i < n; i++)  {

    Sphere sphere = spheres[i]; /* create local copy of sphere */

    /* float hitdistance = intersect_sphere(&spheres[i], ray); */
    float hitdistance = intersect_sphere(&sphere, _ray);
    /* keep track of the closest intersection and hitobject found so far */
    if(hitdistance != 0.0f && hitdistance < *_t) {
      *_t = hitdistance;
      *_id = i;
    }
  }
  return *_t < inf; /* true when ray interesects the scene */
}

static float get_random(unsigned int *_seed0, unsigned int *_seed1)
{
  /* hash the seeds using bitwise AND operations and bitshifts */
  *_seed0 = 36969 * ((*_seed0) & 65535) + ((*_seed0) >> 16);
  *_seed1 = 18000 * ((*_seed1) & 65535) + ((*_seed1) >> 16);

  unsigned int ires = ((*_seed0) << 16) + (*_seed1);

  /* use union struct to convert int to float */
  union {
    float f;
    unsigned int ui;
  } res;

  res.ui = (ires & 0x007fffff) | 0x40000000;  /* bitwise AND, bitwise OR */
  return (res.f - 2.0f) / 2.0f;
}

float3 trace(const Ray *_camray, unsigned int *_seed0, unsigned int *_seed1)
{
  Ray ray = *_camray;

  float3 accum_color = (float3)(0.0f, 0.0f, 0.0f);
  float3 mask = (float3)(1.0f, 1.0f, 1.0f);

  for(int bounces = 0; bounces < 4; bounces++)
  {
    float t;   /* distance to intersection */
    int hitsphere_id = 0; /* index of intersected sphere */

    /* if ray misses scene, return background colour */
    if(!intersect_scene(&ray, &t, &hitsphere_id)) {
      return make_float3(0.f, 0.f, 0.f);
    }

    /* else, we've got a hit! Fetch the closest hit sphere */
    Sphere hitsphere = spheres[hitsphere_id]; /* version with local copy of sphere */
      /* compute the hitpoint using the ray equation */
      float3 hitpoint = ray.m_origin + ray.m_dir * t;

      /* compute the surface normal and flip it if necessary to face the incoming ray */
      float3 normal = normalize(hitpoint - hitsphere.m_pos);
      float3 normal_facing = dot(normal, ray.m_dir) < 0.0f ? normal : normal * (-1.0f);


    if(hitsphere_id == 6) {
      ray.m_origin = hitpoint + normal_facing*0.05f; // offset ray origin slightly to prevent self intersection
      ray.m_dir = ray.m_dir - normal*2*dot(normal, ray.m_dir);
    } else {
      /* compute two random numbers to pick a random point on the hemisphere above the hitpoint*/
      float rand1 = 2.0f * PI * get_random(_seed0, _seed1);
      float rand2 = get_random(_seed0, _seed1);
      float rand2s = sqrt(rand2);

      /* create a local orthogonal coordinate frame centered at the hitpoint */
      float3 w = normal_facing;
      float3 axis = fabs(w.x) > 0.1f ? (float3)(0.0f, 1.0f, 0.0f) : (float3)(1.0f, 0.0f, 0.0f);
      float3 u = normalize(cross(axis, w));
      float3 v = cross(w, u);

      /* use the coordinte frame and random numbers to compute the next ray direction */
      float3 newdir = normalize(u * cos(rand1)*rand2s + v*sin(rand1)*rand2s + w*sqrt(1.0f - rand2));

      /* add a very small offset to the hitpoint to prevent self intersection */
      ray.m_origin = hitpoint + normal_facing * 0.05f;
      ray.m_dir = newdir;

      /* add the colour and light contributions to the accumulated colour */
      accum_color += mask * hitsphere.m_emission;

      /* the mask colour picks up surface colours at each bounce */
      mask *= hitsphere.m_col;

      /* perform cosine-weighted importance sampling for diffuse surfaces*/
      mask *= dot(newdir, normal_facing);
      mask *= 2;
    }
  }

  return accum_color;
}


__kernel void render(__write_only image2d_t _texture, __global float3 *_colors, float3 _cam, float3 _dir, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time)
{
  const unsigned int x = get_global_id(0);
  const unsigned int y = get_global_id(1);

  if(x < _w && y < _h)
  {
    unsigned int ind = y*_w + x;
    unsigned int seed0 = x * _frame;
    unsigned int seed1 = y * _time;
    if(_frame == 1) {
      _colors[ind] = make_float3(0.f, 0.f, 0.f);
    }

    Ray camera = createRay(_cam, _dir);

    float3 cx = make_float3(_w * .5135 / _h, 0.0f, 0.0f); // ray direction offset in x direction
    float3 cy = normalize(cross(cx, camera.m_dir)); // ray direction offset in y direction (.5135 is field of view angle)
    cy.x *= .5135f;
    cy.y *= .5135f;
    cy.z *= .5135f;

//    write_imagef(_texture, (int2)(x, y), make_float4(_tris[ind%1024].x,
//                                                     _tris[ind%1024].y,
//                                                     _tris[ind%1024].z,
//                                                     1.f));

    unsigned int samps = 8;
    for(unsigned int s = 0; s < samps; s++)
    {
      // compute primary ray direction
      float3 d = camera.m_dir + make_float3(cx.x*((.25 + x) / _w - .5),
                                            cx.y*((.25 + x) / _w - .5),
                                            cx.z*((.25 + x) / _w - .5))
                              + make_float3(cy.x*((.25 + y) / _h - .5),
                                            cy.y*((.25 + y) / _h - .5),
                                            cy.z*((.25 + y) / _h - .5));
      // create primary ray, add incoming radiance to pixelcolor
      Ray newcam = createRay(camera.m_origin + d * 40, normalize(d));

      _colors[ind] += trace(&newcam, &seed0, &seed1);
    }
    float coef = 1.f/(samps*_frame);

    write_imagef(_texture, (int2)(x, y), make_float4(pow(clamp(_colors[ind].x * coef, 0.f, 1.f), invGamma),
                                                     pow(clamp(_colors[ind].y * coef, 0.f, 1.f), invGamma),
                                                     pow(clamp(_colors[ind].z * coef, 0.f, 1.f), invGamma),
                                                     1.f));
  }
}
