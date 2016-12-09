#include <iostream>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/device_vector.h>

#include "PathTracer.cuh"
#include "MathHelpers.cuh"

#define invGamma 1.0/2.2
//#define M_PI 3.1415926535

// printf() is only supported
// for devices of compute capability 2.0 and higher
//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
	 #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
//#endif

struct Ray {
	float3 m_origin;
	float3 m_dir;
	__device__ Ray(const float3 &_o, const float3 &_d) : m_origin(_o), m_dir(_d) {}
};

enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()

struct Sphere {
	float m_r;       // radius
	float3 m_pos;
	float3 m_emission;
	float3 m_col;
	Refl_t m_refl;

	__device__ float intersect(const Ray &_r) const
	{ // returns distance, 0 if nohit
		float3 op = m_pos - _r.m_origin; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
		float t;
		float eps = 1e-4;
		float b = dot(op, _r.m_dir);
		float det = b*b - dot(op, op) + m_r*m_r;
		if(det < 0)
			return 0;
		else
			det = sqrtf(det);
		return (t = b-det) > eps ? t : ((t = b+det) > eps ? t : 0.0);
	}
};

__constant__ Sphere spheres[] = {			//Scene: radius, position, emission, color, material
//	 {1e5, { 1e5+1, 40.8, 81.6 }, {},								{.75,.25,.25}, 			DIFF},	//Left
//	 {1e5, {-1e5+99,40.8,81.6}, 	{},								{.25,.25,.75}, 			DIFF},	//Rght
//	 {1e5, {50,40.8, 1e5},      	{},								{.75,.75,.75}, 			DIFF},	//Back
//	 {1e5, {50,40.8,-1e5+170},  	{},								{},           			DIFF},	//Frnt
//	 {1e5, {50, 1e5, 81.6},     	{},								{.75,.75,.75},			DIFF},	//Botm
//	 {1e5, {50,-1e5+81.6,81.6}, 	{},								{.75,.75,.75},			DIFF},	//Top
//	 {16.5,{27,16.5,47},        	{},								{.999, .999, .999},	DIFF},	//Mirr
//	 {16.5,{73,16.5,78},        	{},								{.999, .999, .999},	DIFF},	//Glas
//	 {600, {50,681.6-.27,81.6}, 	{12., 12., 12.},  {}, 								DIFF} 	//Light
	{ 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f }, DIFF }, //Left
	{ 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f }, DIFF }, //Right
	{ 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Back
	{ 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f }, DIFF }, //Frnt
	{ 1e5f, { 50.0f, 1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Botm
	{ 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Top
	{ 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, DIFF }, // small sphere 1
	{ 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, DIFF }, // small sphere 2
	{ 600.0f, { 50.0f, 681.6f - .77f, 81.6f }, { 2.0f, 1.8f, 1.6f }, { 0.0f, 0.0f, 0.0f }, DIFF }  // Light
};

__device__ inline bool intersectScene(const Ray &_r, float &_t, int &_id)
{
	float n = sizeof(spheres)/sizeof(Sphere);
	float d;
	float inf = _t = 1e20;
	for(int i = int(n); i--;) {
		if((d = spheres[i].intersect(_r)) && d < _t)
		{
			_t = d;
			_id = i;
		}
	}
	return _t < inf;
}

__device__ unsigned int randhash(unsigned int a) {
	a = (a+0x7ed55d16) + (a<<12);
	a = (a^0xc761c23c) ^ (a>>19);
	a = (a+0x165667b1) + (a<<5);
	a = (a+0xd3a2646c) ^ (a<<9);
	a = (a+0xfd7046c5) + (a<<3);
	a = (a^0xb55a4f09) ^ (a>>16);
	return a;
}

__device__ static float getrandom(unsigned int *seed0, unsigned int *seed1) {
 *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);  // hash the seeds using bitwise AND and bitshifts
 *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

 unsigned int ires = ((*seed0) << 16) + (*seed1);

 // Convert to float
 union {
	float f;
	unsigned int ui;
 } res;

 res.ui = (ires & 0x007fffff) | 0x40000000;  // bitwise AND, bitwise OR

 return (res.f - 2.f) / 2.f;
}

__device__ float3 radiance(Ray &_r, unsigned int s0, unsigned int s1, unsigned int s2, unsigned int s3)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	float3 color = make_float3(0.0, 0.0, 0.0);
	float3 mask = make_float3(1.0, 1.0, 1.0);

	for(unsigned int bounces = 0; bounces < 4; bounces++)
	{
		float t;																	// distance to intersection
		int id = 0;																// id of intersected object
		if(!intersectScene(_r, t, id)) {
			return make_float3(0.0, 0.0, 0.0);			// if miss, return black
		}

		thrust::default_random_engine rng(randhash(s0)*randhash(s1)*randhash(s2)*randhash(s3));
		thrust::random::uniform_real_distribution<float> uniformDist(0, 1);
		const Sphere &obj = spheres[id];  // hitobject
		float3 x = _r.m_origin + _r.m_dir*t;          // hitpoint
		float3 n = normalize(x - obj.m_pos);    // normal, unsigned int *_s0, unsigned int *_s1
		float3 nl = dot(n, _r.m_dir) < 0 ? n : n * -1; // front facing normal

		color += mask * obj.m_emission;
		float r1 = 2 * M_PI * uniformDist(rng);
		float r2 = uniformDist(rng);
//		float r1 = 2 * M_PI * getrandom(s0, s1); // pick random number on unit circle (radius = 1, circumference = 2*Pi) for azimuth
//		float r2 = getrandom(s0, s1);
		float r2s = sqrtf(r2);
		float3 w = nl;
		float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
		float3 v = cross(w, u);

		float3 d = normalize(( u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1 - r2)));

		_r.m_origin = x + nl*0.05f; // offset ray origin slightly to prevent self intersection
		_r.m_dir = d;

		mask *= obj.m_col;
		mask *= dot(d, nl);
		mask *= 2;
	}
	return color;
}

__global__ void render(cudaSurfaceObject_t _tex, float3 *_colors, float3 _cam, float3 _dir, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < _w && y < _h) {

		unsigned int ind = x + y*_w;
		unsigned int s1 = x;
		unsigned int s2 = y;

//		Ray camera(make_float3(50, 52, 295.6), normalize(make_float3(0, -52/2, -295.6)));
		Ray camera(_cam, _dir);

		float3 cx = make_float3(_w * .5135 / _h, 0.0f, 0.0f); // ray direction offset in x direction
		float3 cy = normalize(cross(cx, camera.m_dir)) * .5135; // ray direction offset in y direction (.5135 is field of view angle)
		float3 color = make_float3(0.0, 0.0, 0.0);

		unsigned int samps = 8;
		for(unsigned int s = 0; s < samps; s++) {  // samples per pixel
			// compute primary ray direction
			float3 d = camera.m_dir + cx*((.25 + x) / _w - .5) + cy*((.25 + y) / _h - .5);
			// create primary ray, add incoming radiance to pixelcolor
			Ray derp(camera.m_origin + d * 40, normalize(d));
			color = color + radiance(derp, x*_frame, ind/_frame, s, _time);
//			color = color + radiance(derp, &s1, &s2, s, _time);
		}
		_colors[ind] += color;

		unsigned char r = (unsigned char)(powf(clamp(_colors[ind].x/(samps*_frame), 0.0, 1.0), invGamma) * 255);
		unsigned char g = (unsigned char)(powf(clamp(_colors[ind].y/(samps*_frame), 0.0, 1.0), invGamma) * 255);
		unsigned char b = (unsigned char)(powf(clamp(_colors[ind].z/(samps*_frame), 0.0, 1.0), invGamma) * 255);

		uchar4 data = make_uchar4(r, g, b, 0xff);
		surf2Dwrite(data, _tex, x*sizeof(uchar4), y);
	}
}

void cu_ModifyTexture(cudaSurfaceObject_t _texture, float3 *_colorArr, float3 _cam, float3 _dir, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time)
{
	dim3 dimBlock(32, 32);
	dim3 dimGrid((_w / dimBlock.x),
							 (_h / dimBlock.y));

	render<<<dimGrid, dimBlock>>>(_texture, _colorArr, _cam, _dir, _w, _h, _frame, _time);
}

void cu_fillFloat3(float3 *d_ptr, float3 _val, unsigned int _size)
{
	thrust::device_ptr<float3> ptr = thrust::device_pointer_cast(d_ptr);
	thrust::fill(ptr, ptr + _size, _val);
}
