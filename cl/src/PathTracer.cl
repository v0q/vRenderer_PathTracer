#include "cl/include/PathTracer.h"
#include "cl/include/RayIntersection.h"

#define BVH_MAX_STACK_SIZE 32
__constant float invGamma = 1.f/2.2f;

__constant Sphere spheres[] = {
		{ 1e5f,		{ 1e5f + 1.0f, 40.8f, 81.6f, 0.f },			{ 0.0f, 0.0f, 0.0f, 0.f },	{ 0.75f, 0.0f, 0.0f, 0.f } }, //Left
		{ 1e5f,		{ -1e5f + 99.0f, 40.8f, 81.6f, 0.f },		{ 0.0f, 0.0f, 0.0f, 0.f },	{ 0.0f, 0.75f, 0.0f, 0.f } }, //Right
		{ 1e5f,		{ 50.0f, 40.8f, 1e5f, 0.f },						{ 0.0f, 0.0f, 0.0f, 0.f },	{ .75f, .75f, .75f, 0.f } }, //Back
		{ 1e5f,		{ 50.0f, 40.8f, -1e5f + 600.0f, 0.f },	{ 0.0f, 0.0f, 0.0f, 0.f },	{ 1.00f, 1.00f, 1.00f, 0.f } }, //Frnt
		{ 1e5f,		{ 50.0f, 1e5f, 81.6f, 0.f },						{ 0.0f, 0.0f, 0.0f, 0.f },	{ .75f, .75f, .75f, 0.f } }, //Botm
		{ 1e5f,		{ 50.0f, -1e5f + 81.6f, 81.6f, 0.f },		{ 0.0f, 0.0f, 0.0f, 0.f },	{ .75f, .75f, .75f, 0.f } }, //Top
//		{ 16.5f,	{ 27.0f, 16.5f, 47.0f, 0.f },						{ 0.0f, 0.0f, 0.0f, 0.f },	{ 1.0f, 1.0f, 1.0f, 0.f } }, // small sphere 1
//		{ 16.5f,	{ 73.0f, 16.5f, 78.0f, 0.f },						{ 0.0f, 0.0f, 0.0f, 0.f },	{ 1.0f, 1.0f, 1.0f, 0.f } }, // small sphere 2
		{ 600.0f, { 50.0f, 681.6f - .77f, 81.6f, 0.f },		{ 2.0f, 1.8f, 1.6f, 0.f },	{ 0.0f, 0.0f, 0.0f, 0.f } }  // Light
};

Ray createRay(float4 _o, float4 _d)
{
	Ray ray;
	ray.m_origin = _o;
  ray.m_dir = _d;
	return ray;
}

bool intersectScene(const Ray *_ray, __read_only image1d_t _triangles, __read_only image1d_t _triIndices, __read_only image1d_t _bvhLimits, __read_only image1d_t _bvhChildrenOrTriangles, vHitData *_hitData)
{
	/* initialise t to a very large number,
	so t will be guaranteed to be smaller
	when a hit with the scene occurs */

	int n = sizeof(spheres)/sizeof(Sphere);;
	float inf = 1e20f;
	float t = inf;

	/* check if the ray intersects each sphere in the scene */
	for(int i = 0; i < n; i++)  {
		/* float hitdistance = intersectSphere(&spheres[i], ray); */
		Sphere sphere = spheres[i]; /* create local copy of sphere */
		float dist = intersectSphere(&sphere, _ray);
		/* keep track of the closest intersection and hitobject found so far */
		if(dist != 0.0f && dist < t) {
			t = dist;
			_hitData->m_hitPoint = _ray->m_origin + _ray->m_dir * t;
			_hitData->m_normal = normalize(_hitData->m_hitPoint - sphere.m_pos);
			_hitData->m_color = sphere.m_col;
			_hitData->m_emission = sphere.m_emission;
		}
  }

  int stackIdx = 0;
  int bvhStack[BVH_MAX_STACK_SIZE];

  bvhStack[stackIdx++] = 0;

  while(stackIdx)
  {
    int bvhIndex = bvhStack[stackIdx - 1];
    uint4 bvhData = read_imageui(_bvhChildrenOrTriangles, (int)(bvhIndex)).xyzw;

    stackIdx--;
    // Inner node
    if(!(bvhData.x & 0x80000000))
    {
      float2 limitsX = read_imagef(_bvhLimits, (int)(3 * bvhIndex)).xy;
      float2 limitsY = read_imagef(_bvhLimits, (int)(3 * bvhIndex + 1)).xy;
      float2 limitsZ = read_imagef(_bvhLimits, (int)(3 * bvhIndex + 2)).xy;

      float3 bottom = (float3)(limitsX.x, limitsY.x, limitsZ.x);
      float3 top = (float3)(limitsX.y, limitsY.y, limitsZ.y);
      if(intersectCFBVH(_ray, bottom, top))
      {
        bvhStack[stackIdx++] = bvhData.y;
        bvhStack[stackIdx++] = bvhData.z;

        if(stackIdx > BVH_MAX_STACK_SIZE)
          return false;
      }
    }
    else
    {
      for(unsigned int i = bvhData.w; i < bvhData.w + (bvhData.x ^ 0x80000000); ++i)
      {
        unsigned int triIndex = read_imageui(_triIndices, (int)(i)).x;

        float4 center = read_imagef(_triangles, (int)(5 * triIndex));
        float4 normal = read_imagef(_triangles, (int)(5 * triIndex + 1));

        float k = dot(normal, _ray->m_dir);
        if(k == 0.f)
          continue;

        float s = (normal.w - dot(normal, _ray->m_origin)) / k;
        if(s <= epsilon)
          continue;

        float4 hit = _ray->m_dir * s;
        hit += _ray->m_origin;

        float4 ee1 = read_imagef(_triangles, (int)(5 * triIndex + 2));
        float kt1 = dot(ee1, hit) - ee1.w;
        if(kt1 < epsilon)
          continue;

        float4 ee2 = read_imagef(_triangles, (int)(5 * triIndex + 3));
        float kt2 = dot(ee2, hit) - ee2.w;
        if(kt2 < epsilon)
          continue;

        float4 ee3 = read_imagef(_triangles, (int)(5 * triIndex + 4));
        float kt3 = dot(ee3, hit) - ee3.w;
        if(kt3 < epsilon)
          continue;

        float distSquared = dot(_ray->m_origin - hit, _ray->m_origin - hit);
        if(distSquared < t * t)
        {
          t = length(_ray->m_origin - hit);

          _hitData->m_hitPoint = _ray->m_origin + _ray->m_dir * t;
          _hitData->m_normal = normal;
          _hitData->m_color = (float4)(1.0f, 1.0f, 1.0f, 0.f);
          _hitData->m_emission = (float4)(0.0f, 0.0f, 0.0f, 0.f);
        }
      }
    }
  }


	return t < inf; /* true when ray interesects the scene */
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

float4 trace(const Ray *_camray, __read_only image1d_t _triangles, __read_only image1d_t _triIndices, __read_only image1d_t _bvhLimits, __read_only image1d_t _bvhChildrenOrTriangles, unsigned int *_seed0, unsigned int *_seed1)
{
	Ray ray = *_camray;

	float4 accum_color = (float4)(0.0f, 0.0f, 0.0f, 0.f);
	float4 mask = (float4)(1.0f, 1.0f, 1.0f, 0.f);

	for(int bounces = 0; bounces < 4; bounces++)
	{
		vHitData hitData;

		/* if ray misses scene, return background colour */
    if(!intersectScene(&ray, _triangles, _triIndices, _bvhLimits, _bvhChildrenOrTriangles, &hitData)) {
			return (float4)(0.f, 0.f, 0.f, 0.f);
		}

		/* compute the surface normal and flip it if necessary to face the incoming ray */
		float4 normal_facing = dot(hitData.m_normal, ray.m_dir) < 0.0f ? hitData.m_normal : hitData.m_normal * (-1.0f);

    /* compute two random numbers to pick a random point on the hemisphere above the hitpoint*/
    float rand1 = 2.0f * PI * get_random(_seed0, _seed1);
    float rand2 = get_random(_seed0, _seed1);
    float rand2s = sqrt(rand2);

    /* create a local orthogonal coordinate frame centered at the hitpoint */
    float4 w = normal_facing;
    float4 axis = fabs(w.x) > 0.1f ? (float4)(0.0f, 1.0f, 0.0f, 0.f) : (float4)(1.0f, 0.0f, 0.0f, 0.f);
    float4 u = normalize(cross(axis, w));
    float4 v = cross(w, u);

    /* use the coordinte frame and random numbers to compute the next ray direction */
    float4 newdir = normalize(u * cos(rand1)*rand2s + v*sin(rand1)*rand2s + w*sqrt(1.0f - rand2));

    /* add a very small offset to the hitpoint to prevent self intersection */
    ray.m_origin = hitData.m_hitPoint + normal_facing * 0.05f;
    ray.m_dir = newdir;

    /* add the colour and light contributions to the accumulated colour */
    accum_color += mask * hitData.m_emission;

    /* the mask colour picks up surface colours at each bounce */
    mask *= hitData.m_color;

    /* perform cosine-weighted importance sampling for diffuse surfaces*/
    mask *= dot(newdir, normal_facing);
    mask *= 2;
	}

	return accum_color;
}


__kernel void render(__write_only image2d_t _texture, __read_only image1d_t _triangles, __read_only image1d_t _triIndices, __read_only image1d_t _bvhLimits, __read_only image1d_t _bvhChildrenOrTriangles,
                     __global float4 *_colors, float4 _cam, float4 _dir, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time)
{
	const unsigned int x = get_global_id(0);
	const unsigned int y = get_global_id(1);

	if(x < _w && y < _h)
	{
		unsigned int ind = y*_w + x;
		unsigned int seed0 = x * _frame;
		unsigned int seed1 = y * _time;
		if(_frame == 1) {
			_colors[ind] = (float4)(0.f, 0.f, 0.f, 0.f);
		}

		Ray camera = createRay(_cam, _dir);

		float4 cx = (float4)(_w * .5135 / _h, 0.0f, 0.0f, 0.f); // ray direction offset in x direction
		float4 cy = normalize(cross(cx, camera.m_dir)); // ray direction offset in y direction (.5135 is field of view angle)
		cy.x *= .5135f;
		cy.y *= .5135f;
		cy.z *= .5135f;

    unsigned int samps = 1;
		for(unsigned int s = 0; s < samps; s++)
		{
			// compute primary ray direction
			float4 d = camera.m_dir + (float4)(cx.x*((.25 + x) / _w - .5),
																				 cx.y*((.25 + x) / _w - .5),
																				 cx.z*((.25 + x) / _w - .5), 0.f)
															+ (float4)(cy.x*((.25 + y) / _h - .5),
																				 cy.y*((.25 + y) / _h - .5),
																				 cy.z*((.25 + y) / _h - .5), 0.f);
			// create primary ray, add incoming radiance to pixelcolor
			Ray newcam = createRay(camera.m_origin + d * 40, normalize(d));

      _colors[ind] += trace(&newcam, _triangles, _triIndices, _bvhLimits, _bvhChildrenOrTriangles, &seed0, &seed1);
		}
		float coef = 1.f/(samps*_frame);

		write_imagef(_texture, (int2)(x, y), (float4)(pow(clamp(_colors[ind].x * coef, 0.f, 1.f), invGamma),
																									pow(clamp(_colors[ind].y * coef, 0.f, 1.f), invGamma),
																									pow(clamp(_colors[ind].z * coef, 0.f, 1.f), invGamma),
																									1.f));
	}
}
