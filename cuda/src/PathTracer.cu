#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/device_vector.h>

#include "PathTracer.cuh"
#include "RayIntersection.cuh"
#include "MathHelpers.cuh"

#define BVH_MAX_STACK_SIZE 32

__constant__ __device__ uint bvhBoxes = 0;
__constant__ __device__ uint kSamps = 1;
__constant__ __device__ float kInvGamma = 1.f/2.2f;
__constant__ __device__ float kInvSamps = 1.f/1.f;

texture<uint1, 1, cudaReadModeElementType> t_triIndices;
texture<float2, 1, cudaReadModeElementType> t_bvhLimits;
texture<uint4, 1, cudaReadModeElementType> t_bvhChildrenOrTriangles;
texture<float4, 1, cudaReadModeElementType> t_triangles;

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
//	{ 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f, 0.0f },			{ 0.075f, 0.f, 0.f, 0.0f }, { 0.75f, 0.0f, 0.0f, 0.0f } }, //Left
//	{ 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f, 0.0f },		{ 0.f, 0.075f, 0.f, 0.0f }, { 0.0f, 0.75f, 0.0f, 0.0f } }, //Right
//	{ 1e5f, { 50.0f, 40.8f, 1e5f, 0.0f },							{ 0.0f, 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f, 0.0f } }, //Back
//	{ 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f, 0.0f },		{ 0.0f, 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f, 0.0f } }, //Frnt
	{ 1e5f, { 50.0f, 1e5f, 81.6f, 0.0f },							{ 0.0f, 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f, 0.0f } }, //Botm
//	{ 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f, 0.0f },		{ 0.0f, 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f, 0.0f } }, //Top
//	{ 16.5f, { 27.0f, 16.5f, 47.0f, 0.0f },						{ 0.0f, 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f, 0.0f } }, // small sphere 1
//	{ 16.5f, { 73.0f, 16.5f, 78.0f, 0.0f },						{ 0.0f, 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f, 0.0f } }, // small sphere 2
	{ 150.0f, { 50.0f, 300.6f - .77f, 81.6f, 0.0f },	{ 2.0f, 1.8f, 1.6f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } }  // Light
};

//__device__ inline bool intersectScene(const Ray *_ray, float4 *_triangleData, unsigned int *_triIdxList, float2 *_bvhLimits, uint4 *_bvhChildrenOrTriangles, vHitData *_hitData)
__device__ inline bool intersectScene(const Ray *_ray, vHitData *_hitData)
{
	/* initialise t to a very large number,
	so t will be guaranteed to be smaller
	when a hit with the scene occurs */

	int n = sizeof(spheres)/sizeof(Sphere);
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

	int stackIdx = 0;
	int bvhStack[BVH_MAX_STACK_SIZE];

	bvhStack[stackIdx++] = 0;

	while(stackIdx)
	{
		int bvhIndex = bvhStack[stackIdx - 1];
		uint4 bvhData = tex1Dfetch(t_bvhChildrenOrTriangles, bvhIndex);
//		uint4 bvhData = _bvhChildrenOrTriangles[bvhIndex];

		stackIdx--;
		// Inner node
		if(!(bvhData.x & 0x80000000))
		{
			float2 limitsX = tex1Dfetch(t_bvhLimits, 3 * bvhIndex);
			float2 limitsY = tex1Dfetch(t_bvhLimits, 3 * bvhIndex + 1);
			float2 limitsZ = tex1Dfetch(t_bvhLimits, 3 * bvhIndex + 2);
//			float2 limitsX = _bvhLimits[3 * bvhIndex];
//			float2 limitsY = _bvhLimits[3 * bvhIndex + 1];
//			float2 limitsZ = _bvhLimits[3 * bvhIndex + 2];

			float3 bottom = make_float3(limitsX.x, limitsY.x, limitsZ.x);
			float3 top = make_float3(limitsX.y, limitsY.y, limitsZ.y);
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
				unsigned int triIndex = tex1Dfetch(t_triIndices, i).x;
//				unsigned int triIndex = _triIdxList[i];

				float4 center = tex1Dfetch(t_triangles, 5 * triIndex);
				float4 normal = tex1Dfetch(t_triangles, 5 * triIndex + 1);
//				float4 center = _triangleData[5 * triIndex];
//				float4 normal = _triangleData[5 * triIndex + 1];

				float k = dot(normal, _ray->m_dir);
				if(k == 0.f)
					continue;

				float s = (normal.w - dot(normal, _ray->m_origin)) / k;
				if(s <= epsilon)
					continue;

				float4 hit = _ray->m_dir * s;
				hit += _ray->m_origin;

//				float4 ee1 = _triangleData[5 * triIndex + 2];
				float4 ee1 = tex1Dfetch(t_triangles, 5 * triIndex + 2);
				float kt1 = dot(ee1, hit) - ee1.w;
				if(kt1 < epsilon)
					continue;

//				float4 ee2 = _triangleData[5 * triIndex + 3];
				float4 ee2 = tex1Dfetch(t_triangles, 5 * triIndex + 3);
				float kt2 = dot(ee2, hit) - ee2.w;
				if(kt2 < epsilon)
					continue;

//				float4 ee3 = _triangleData[5 * triIndex + 4];
				float4 ee3 = tex1Dfetch(t_triangles, 5 * triIndex + 4);
				float kt3 = dot(ee3, hit) - ee3.w;
				if(kt3 < epsilon)
					continue;

				float distSquared = distanceSquared(_ray->m_origin - hit);
				if(distSquared < t * t) {
					t = distance(_ray->m_origin - hit);
					_hitData->m_hitPoint = hit;
					_hitData->m_color = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
					_hitData->m_normal = normal;
					_hitData->m_emission = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				}
			}
		}
	}

	return t < inf; /* true when ray interesects the scene */
}

__device__ static unsigned int hash(unsigned int *seed0, unsigned int *seed1)
{
 *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16); // hash the seeds using bitwise AND and bitshifts
 *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

	return *seed0**seed1;
}

//__device__ float4 trace(const Ray *_camray, float4 *_triangleData, unsigned int *_triIdxList, float2 *_bvhLimits, uint4 *_bvhChildrenOrTriangles, unsigned int *_seed0, unsigned int *_seed1)
__device__ float4 trace(const Ray *_camray, unsigned int *_seed0, unsigned int *_seed1)
{
	Ray ray = *_camray;

	float4 accum_color = make_float4(0.0f, 0.0f, 0.0f, 0.f);
	float4 mask = make_float4(1.0f, 1.0f, 1.0f, 0.f);

	for(unsigned int bounces = 0; bounces < 5; bounces++)
	{
		vHitData hitData;

//		if(!intersectScene(&ray, _triangleData, _triIdxList, _bvhLimits, _bvhChildrenOrTriangles, &hitData)) {
		if(!intersectScene(&ray, &hitData)) {
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

//__global__ void render(cudaSurfaceObject_t _tex, float4 *_triangleData, unsigned int *_triIdxList, float2 *_bvhLimits, uint4 *_bvhChildrenOrTriangles,
//											 float4 *_colors, float4 *_cam, float4 *_dir, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time)
__global__ void render(cudaSurfaceObject_t _tex, float4 *_colors, float4 *_cam, float4 *_dir, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time)
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

		for(unsigned int s = 0; s < kSamps; s++) {  // samples per pixel
			// compute primary ray direction
			float4 d = camera.m_dir + cx*((.25 + x) / _w - .5) + cy*((.25 + y) / _h - .5);
			// create primary ray, add incoming radiance to pixelcolor
			Ray newcam(camera.m_origin + d * 40, normalize(d));
//			_colors[ind] += trace(&newcam, _triangleData, _triIdxList, _bvhLimits, _bvhChildrenOrTriangles, &s1, &s2) * (kInvSamps);
			_colors[ind] += trace(&newcam, &s1, &s2) * (kInvSamps);
		}

		float coef = 1.f/_frame;
		unsigned char r = (unsigned char)(powf(clamp(_colors[ind].x*coef, 0.0, 1.0), kInvGamma) * 255);
		unsigned char g = (unsigned char)(powf(clamp(_colors[ind].y*coef, 0.0, 1.0), kInvGamma) * 255);
		unsigned char b = (unsigned char)(powf(clamp(_colors[ind].z*coef, 0.0, 1.0), kInvGamma) * 255);

		uchar4 data = make_uchar4(r, g, b, 0xff);
		surf2Dwrite(data, _tex, x*sizeof(uchar4), y);
	}
}

void cu_runRenderKernel(// Buffers
												cudaSurfaceObject_t _texture, float4 *_triangleData, unsigned int *_triIdxList, float2 *_bvhLimits, uint4 *_bvhChildrenOrTriangles,
												// Buffer sizes for texture initialisation
												unsigned int _triCount, unsigned int _bvhBoxCount, unsigned int _triIdxCount,
												float4 *_colorArr, float4 *_cam, float4 *_dir,
												unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time)
{
	static bool m_initialised = false;
	if(!m_initialised)
	{
		// bind the scene data to CUDA textures!
		m_initialised = true;

		cudaChannelFormatDesc channel1desc = cudaCreateChannelDesc<uint1>();
		cudaBindTexture(NULL, &t_triIndices, _triIdxList, &channel1desc, _triIdxCount * sizeof(uint1));

		cudaChannelFormatDesc channel2desc = cudaCreateChannelDesc<uint4>();
		cudaBindTexture(NULL, &t_bvhChildrenOrTriangles, _bvhChildrenOrTriangles, &channel2desc, _bvhBoxCount * sizeof(uint4));

		cudaChannelFormatDesc channel3desc = cudaCreateChannelDesc<float2>();
		cudaBindTexture(NULL, &t_bvhLimits, _bvhLimits, &channel3desc, _bvhBoxCount * 3 * sizeof(float2));

		cudaChannelFormatDesc channel5desc = cudaCreateChannelDesc<float4>();
		cudaBindTexture(NULL, &t_triangles, _triangleData, &channel5desc, _triCount * 5 * sizeof(float4));
	}

	dim3 dimBlock(16, 16);
	dim3 dimGrid((_w / dimBlock.x),
							 (_h / dimBlock.y));

//	render<<<dimGrid, dimBlock>>>(_texture, _triangleData, _triIdxList, _bvhLimits, _bvhChildrenOrTriangles, _colorArr, _cam, _dir, _w, _h, _frame, _time);
	render<<<dimGrid, dimBlock>>>(_texture, _colorArr, _cam, _dir, _w, _h, _frame, _time);
}

void cu_updateBVHBoxCount(unsigned int _bvhBoxes)
{
	cudaMemcpyToSymbol(bvhBoxes, &_bvhBoxes, sizeof(unsigned int));
}

void cu_fillFloat4(float4 *d_ptr, float4 _val, unsigned int _size)
{
	thrust::device_ptr<float4> ptr = thrust::device_pointer_cast(d_ptr);
	thrust::fill(ptr, ptr + _size, _val);
}
