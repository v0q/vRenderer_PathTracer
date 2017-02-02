#include <iostream>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/device_vector.h>

#include "PathTracer.cuh"
#include "MathHelpers.cuh"

#define invGamma 1.f/2.2f
#define PI 3.14159265359f
#define EPSILON 0.0000003f

typedef struct Ray {
	float4 m_origin;
	float4 m_dir;
	__device__ Ray(float4 _o, float4 _d) : m_origin(_o), m_dir(_d) {}
} Ray;

typedef struct Sphere {
	float m_r;       // radius
	float4 m_pos;
	float4 m_emission;
	float4 m_col;

	__device__ float intersect(const Ray *_r) const
	{ // returns distance, 0 if nohit
		float4 op = m_pos - _r->m_origin; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
		float t;
		float eps = 1e-4;
		float b = dot(op, _r->m_dir);
		float det = b*b - dot(op, op) + m_r*m_r;
		if(det < 0)
			return 0;
		else
			det = sqrtf(det);
		return (t = b-det) > eps ? t : ((t = b+det) > eps ? t : 0.0);
	}
} Sphere;

__constant__ Sphere spheres[] = {			//Scene: radius, position, emission, color, material
	{ 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f, 0.0f },			{ 0.075f, 0.f, 0.f, 0.0f }, { 0.75f, 0.0f, 0.0f, 0.0f } }, //Left
	{ 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f, 0.0f },		{ 0.f, 0.075f, 0.f, 0.0f }, { 0.0f, 0.75f, 0.0f, 0.0f } }, //Right
	{ 1e5f, { 50.0f, 40.8f, 1e5f, 0.0f },							{ 0.0f, 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f, 0.0f } }, //Back
	{ 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f, 0.0f },		{ 0.0f, 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f, 0.0f } }, //Frnt
	{ 1e5f, { 50.0f, 1e5f, 81.6f, 0.0f },							{ 0.0f, 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f, 0.0f } }, //Botm
	{ 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f, 0.0f },		{ 0.0f, 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f, 0.0f } }, //Top
	{ 16.5f, { 27.0f, 16.5f, 47.0f, 0.0f },						{ 0.0f, 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f, 0.0f } }, // small sphere 1
	{ 16.5f, { 73.0f, 16.5f, 78.0f, 0.0f },						{ 0.0f, 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f, 0.0f } }, // small sphere 2
	{ 600.0f, { 50.0f, 681.6f - .77f, 81.6f, 0.0f },	{ 2.0f, 1.8f, 1.6f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } }  // Light
};

__device__ float intersectTriangle(const float4 &_v1, const float4 &_v2, const float4 &_v3, const Ray *_ray)
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

__device__ inline bool intersectScene(const Ray *_ray,  const vTriangle *_scene, unsigned int _triCount, vHitData *_hitData)
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
		float dist = sphere.intersect(_ray);
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
			_hitData->m_color = make_float4(1.f, 1.f, 1.f, 0.f);
			_hitData->m_emission = make_float4(0.f, 0.0f, 0.0f, 0.f);
		}
	}

	return t < inf; /* true when ray interesects the scene */
}

__device__ static unsigned int hash(unsigned int *seed0, unsigned int *seed1) {
 *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);  // hash the seeds using bitwise AND and bitshifts
 *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

	return *seed0**seed1;
}

__device__ float4 trace(const Ray *_camray, const vTriangle *_scene, unsigned int _triCount, unsigned int *_seed0, unsigned int *_seed1)
{
	Ray ray = *_camray;

	float4 accum_color = make_float4(0.0f, 0.0f, 0.0f, 0.f);
	float4 mask = make_float4(1.0f, 1.0f, 1.0f, 0.f);

	for(unsigned int bounces = 0; bounces < 4; bounces++)
	{
		vHitData hitData;

		if(!intersectScene(&ray, _scene, _triCount, &hitData)) {
			return make_float4(0.f, 0.f, 0.f, 0.f);
		}

		unsigned int seed = hash(_seed0, _seed1);
		thrust::default_random_engine rng(seed);
		thrust::random::uniform_real_distribution<float> uniformDist(0, 1);

		/* compute the surface normal and flip it if necessary to face the incoming ray */
		float4 normal_facing = dot(hitData.m_normal, ray.m_dir) < 0.0f ? hitData.m_normal : hitData.m_normal * (-1.0f);
		/* compute two random numbers to pick a random point on the hemisphere above the hitpoint*/
		float rand1 = 2.0f * PI * uniformDist(rng);
		float rand2 = uniformDist(rng);
		float rand2s = sqrt(rand2);

		/* create a local orthogonal coordinate frame centered at the hitpoint */
		float4 w = normal_facing;
		float4 axis = fabs(w.x) > 0.1f ? make_float4(0.0f, 1.0f, 0.0f, 0.f) : make_float4(1.0f, 0.0f, 0.0f, 0.f);
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

__global__ void render(cudaSurfaceObject_t _tex, const vTriangle *_scene, unsigned int _triCount, float4 *_colors, float4 *_cam, float4 *_dir, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < _w && y < _h) {

		unsigned int ind = x + y*_w;
    unsigned int s1 = x * _frame;
		unsigned int s2 = y * _time;

    if(_frame == 1) {
			_colors[ind] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

		Ray camera(*_cam, *_dir);

		float4 cx = make_float4(_w * .5135 / _h, 0.0f, 0.0f, 0.0f); // ray direction offset in x direction
		float4 cy = normalize(cross(cx, camera.m_dir)) * .5135; // ray direction offset in y direction (.5135 is field of view angle)

		unsigned int samps = 8;
		for(unsigned int s = 0; s < samps; s++) {  // samples per pixel
			// compute primary ray direction
			float4 d = camera.m_dir + cx*((.25 + x) / _w - .5) + cy*((.25 + y) / _h - .5);
			// create primary ray, add incoming radiance to pixelcolor
			Ray newcam(camera.m_origin + d * 40, normalize(d));
			_colors[ind] += trace(&newcam, _scene, _triCount, &s1, &s2);
		}

		float coef = 1.f/(samps*_frame);
		unsigned char r = (unsigned char)(powf(clamp(_colors[ind].x*coef, 0.0, 1.0), invGamma) * 255);
		unsigned char g = (unsigned char)(powf(clamp(_colors[ind].y*coef, 0.0, 1.0), invGamma) * 255);
		unsigned char b = (unsigned char)(powf(clamp(_colors[ind].z*coef, 0.0, 1.0), invGamma) * 255);

		uchar4 data = make_uchar4(r, g, b, 0xff);
		surf2Dwrite(data, _tex, x*sizeof(uchar4), y);
	}
}

void cu_runRenderKernel(cudaSurfaceObject_t _texture, const vTriangle *_scene, unsigned int _triCount, float4 *_colorArr, float4 *_cam, float4 *_dir, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time)
{
	dim3 dimBlock(16, 16);
	dim3 dimGrid((_w / dimBlock.x),
							 (_h / dimBlock.y));

	render<<<dimGrid, dimBlock>>>(_texture, _scene, _triCount, _colorArr, _cam, _dir, _w, _h, _frame, _time);
}

void cu_fillFloat4(float4 *d_ptr, float4 _val, unsigned int _size)
{
	thrust::device_ptr<float4> ptr = thrust::device_pointer_cast(d_ptr);
	thrust::fill(ptr, ptr + _size, _val);
}
