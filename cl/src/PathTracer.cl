#include "cl/include/PathTracer.h"

__constant float invGamma = 1.f/2.2f;
__constant float PI = 3.14159265359f;
__constant float EPSILON = 0.0000003f;
__constant Sphere spheres[] = {
		{ 1e5f,		{ 1e5f + 1.0f, 40.8f, 81.6f, 0.f },			{ 0.0f, 0.0f, 0.0f, 0.f },	{ 0.75f, 0.0f, 0.0f, 0.f } }, //Left
		{ 1e5f,		{ -1e5f + 99.0f, 40.8f, 81.6f, 0.f },		{ 0.0f, 0.0f, 0.0f, 0.f },	{ 0.0f, 0.75f, 0.0f, 0.f } }, //Right
		{ 1e5f,		{ 50.0f, 40.8f, 1e5f, 0.f },						{ 0.0f, 0.0f, 0.0f, 0.f },	{ .75f, .75f, .75f, 0.f } }, //Back
		{ 1e5f,		{ 50.0f, 40.8f, -1e5f + 600.0f, 0.f },	{ 0.0f, 0.0f, 0.0f, 0.f },	{ 1.00f, 1.00f, 1.00f, 0.f } }, //Frnt
		{ 1e5f,		{ 50.0f, 1e5f, 81.6f, 0.f },						{ 0.0f, 0.0f, 0.0f, 0.f },	{ .75f, .75f, .75f, 0.f } }, //Botm
		{ 1e5f,		{ 50.0f, -1e5f + 81.6f, 81.6f, 0.f },		{ 0.0f, 0.0f, 0.0f, 0.f },	{ .75f, .75f, .75f, 0.f } }, //Top
		{ 16.5f,	{ 27.0f, 16.5f, 47.0f, 0.f },						{ 0.0f, 0.0f, 0.0f, 0.f },	{ 1.0f, 1.0f, 1.0f, 0.f } }, // small sphere 1
		{ 16.5f,	{ 73.0f, 16.5f, 78.0f, 0.f },						{ 0.0f, 0.0f, 0.0f, 0.f },	{ 1.0f, 1.0f, 1.0f, 0.f } }, // small sphere 2
		{ 600.0f, { 50.0f, 681.6f - .77f, 81.6f, 0.f },		{ 2.0f, 1.8f, 1.6f, 0.f },	{ 0.0f, 0.0f, 0.0f, 0.f } }  // Light
};

Ray createRay(float4 _o, float4 _d)
{
	Ray ray;
	ray.m_origin = _o;
	ray.m_dir = _d;
	return ray;
}

float intersectTriangle(const float4 _v1, const float4 _v2, const float4 _v3, const Ray *_ray)
{
	float4 e1, e2;  //Edge1, Edge2
	float4 p, q, t;
	float det, inv_det, u, v;
	float dist;

	//Find vectors for two edges sharing V1
	e1 = _v2 - _v1;
	e2 = _v3 - _v1;
	//Begin calculating determinant - also used to calculate u parameter

	p = cross(_ray->m_dir, e2);
	//if determinant is near zero, ray lies in plane of triangle or ray is parallel to plane of triangle
	det = dot(e1, p);
	//NOT CULLING
	if(det > -EPSILON && det < EPSILON)
		return 0.f;
	inv_det = 1.f / det;

	//calculate distance from V1 to ray origin
	t = _ray->m_origin - _v1;

	//Calculate u parameter and test bound
	u = dot(t, p) * inv_det;
	//The intersection lies outside of the triangle
	if(u < 0.f || u > 1.f)
		return 0.f;

	//Prepare to test v parameter
	q = cross(t, e1);

	//Calculate V parameter and test bound
	v = dot(_ray->m_dir, q) * inv_det;
	//The intersection lies outside of the triangle
	if(v < 0.f || u + v  > 1.f)
		return 0.f;

	dist = dot(e2, q) * inv_det;

	if(dist > EPSILON)
		return dist;

	// No hit, no win
	return 0.f;
}

float intersectSphere(const Sphere *_sphere, const Ray *_ray) /* version using local copy of sphere */
{
	float4 rayToCenter = _sphere->m_pos - _ray->m_origin;
	float b = dot(rayToCenter, _ray->m_dir);
	float c = dot(rayToCenter, rayToCenter) - _sphere->m_r*_sphere->m_r;
	float disc = b * b - c;

	if (disc < 0.0f) return 0.0f;
	else disc = sqrt(disc);

	if ((b - disc) > EPSILON) return b - disc;
	if ((b + disc) > EPSILON) return b + disc;

	return 0.0f;
}

bool intersectScene(const Ray *_ray, __global const vTriangle *_scene, unsigned int _triCount, vHitData *_hitData)
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
	for(unsigned int i = 0; i < _triCount; ++i)
	{
		float dist = intersectTriangle(_scene[i].m_v1.m_vert, _scene[i].m_v2.m_vert, _scene[i].m_v3.m_vert, _ray);
		if(dist != 0.0f && dist < t) {
			t = dist;
			_hitData->m_hitPoint = _ray->m_origin + _ray->m_dir * t;
			_hitData->m_normal = _scene[i].m_v1.m_normal;
			_hitData->m_color = (float4)(1.f, 1.f, 1.f, 0.f);
			_hitData->m_emission = (float4)(0.f, 0.0f, 0.0f, 0.f);
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

float4 trace(const Ray *_camray, __global const vTriangle *_scene, unsigned int _triCount, unsigned int *_seed0, unsigned int *_seed1)
{
	Ray ray = *_camray;

	float4 accum_color = (float4)(0.0f, 0.0f, 0.0f, 0.f);
	float4 mask = (float4)(1.0f, 1.0f, 1.0f, 0.f);

	for(int bounces = 0; bounces < 4; bounces++)
	{
		vHitData hitData;

		/* if ray misses scene, return background colour */
		if(!intersectScene(&ray, _scene, _triCount, &hitData)) {
			return (float4)(0.f, 0.f, 0.f, 0.f);
		}

		/* compute the surface normal and flip it if necessary to face the incoming ray */
		float4 normal_facing = dot(hitData.m_normal, ray.m_dir) < 0.0f ? hitData.m_normal : hitData.m_normal * (-1.0f);

//    if(hitsphere_id == 6) {
//      ray.m_origin = hitpoint + normal_facing*0.05f; // offset ray origin slightly to prevent self intersection
//      ray.m_dir = ray.m_dir - hitData.m_normal*2*dot(hitData.m_normal, ray.m_dir);
//    } else {
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
//    }
	}

	return accum_color;
}


__kernel void render(__write_only image2d_t _texture, __global const vTriangle *_scene, unsigned int _triCount, __global float4 *_colors, float4 _cam, float4 _dir, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time)
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

		unsigned int samps = 8;
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

			_colors[ind] += trace(&newcam, _scene, _triCount, &seed0, &seed1);
		}
		float coef = 1.f/(samps*_frame);

		write_imagef(_texture, (int2)(x, y), (float4)(pow(clamp(_colors[ind].x * coef, 0.f, 1.f), invGamma),
																									pow(clamp(_colors[ind].y * coef, 0.f, 1.f), invGamma),
																									pow(clamp(_colors[ind].z * coef, 0.f, 1.f), invGamma),
																									1.f));
	}
}
