#include <stdio.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/device_vector.h>

#include "PathTracer.cuh"
#include "RayIntersection.cuh"
#include "MathHelpers.cuh"

#define BVH_MAX_STACK_SIZE 32

__constant__ __device__ uint bvhBoxes = 0;
__constant__ __device__ uint kSamps = 2;
__constant__ __device__ float kInvGamma = 1.f/2.2f;
__constant__ __device__ float kInvSamps = 1.f/2.f;
__constant__ __device__ float kFov = 75.f * 3.1415 / 180.f;
__constant__ __device__ unsigned int kHDRwidth = 0;
__constant__ __device__ unsigned int kHDRheight = 0;

texture<float4, 1, cudaReadModeElementType> t_hdr;

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
	{ 1e5f, { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f, 0.0f } }, //Botm
//	{ 16.5f, { 27.0f, 16.5f, 47.0f, 0.0f },						{ 0.0f, 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f, 0.0f } }, // small sphere 1
//	{ 16.5f, { 73.0f, 16.5f, 78.0f, 0.0f },						{ 0.0f, 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f, 0.0f } }, // small sphere 2
//	{ 150.0f, { 50.0f, 300.6f - .77f, 81.6f, 0.0f },	/*{ 2.0f, 1.8f, 1.6f, 0.0f }*/{ 2.8f, 1.8f, 1.6f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } }  // Light
};

__device__ __inline__ void swap(int &_a, int &_b)
{
	int tmp = _a;
	_a = _b;
	_b = tmp;
}

__device__ inline bool intersectScene(const Ray *_ray, float4 *_vertices, float4 *_normals, float4 *_bvhData, vHitData *_hitData)
{
	/* initialise t to a very large number,
	so t will be guaranteed to be smaller
	when a hit with the scene occurs */

	int n = sizeof(spheres)/sizeof(Sphere);
	float inf = 1e20f;
	float t = inf;

//	/* check if the ray intersects each sphere in the scene */
//	for(int i = 0; i < n; i++)  {
//		/* float hitdistance = intersectSphere(&spheres[i], ray); */
//		Sphere sphere = spheres[i]; /* create local copy of sphere */
//		float dist = sphere.intersect(_ray);
//		/* keep track of the closest intersection and hitobject found so far */
//		if(dist != 0.0f && dist < t) {
//			t = dist;
//			_hitData->m_hitPoint = _ray->m_origin + _ray->m_dir * t;
//			_hitData->m_normal = normalize(_hitData->m_hitPoint - sphere.m_pos);
//			_hitData->m_color = sphere.m_col;
//			_hitData->m_emission = sphere.m_emission;
//		}
//	}

	const int EntrypointSentinel = 0x76543210;
	int startNode = 0;
	int traversalStack[64];
	traversalStack[0] = EntrypointSentinel;

	char* stackPtr;                       // Current position in traversal stack.
	int leafAddr;                       // First postponed leaf, non-negative if none.
	int nodeAddr = EntrypointSentinel;  // Non-negative: current internal node, negative: second postponed leaf.
	stackPtr = (char*)&traversalStack[0];
	leafAddr = 0;   // No postponed leaf.
	nodeAddr = startNode;   // Start from the root.


	float3 invDir = make_float3(1.0f / (fabsf(_ray->m_dir.x) > epsilon ? _ray->m_dir.x : epsilon),
															1.0f / (fabsf(_ray->m_dir.y) > epsilon ? _ray->m_dir.y : epsilon),
															1.0f / (fabsf(_ray->m_dir.z) > epsilon ? _ray->m_dir.z : epsilon));
	float3 od = make_float3(_ray->m_origin.x * invDir.x,
													_ray->m_origin.y * invDir.y,
													_ray->m_origin.z * invDir.z);

	while(nodeAddr != EntrypointSentinel)
	{
		while((unsigned int)nodeAddr < (unsigned int)EntrypointSentinel)
		{
//			const float4 n0xy = tex1Dfetch(t_bvhData, nodeAddr + 0); // node 0 bounds xy
//			const float4 n1xy = tex1Dfetch(t_bvhData, nodeAddr + 1); // node 1 bounds xy
//			const float4 nz = tex1Dfetch(t_bvhData, nodeAddr + 2); // node 0 & 1 bounds z
//			float4 tmp = tex1Dfetch(t_bvhData, nodeAddr + 3); // Child indices in x & y
			const float4 n0xy = _bvhData[nodeAddr + 0]; // node 0 bounds xy
			const float4 n1xy = _bvhData[nodeAddr + 1]; // node 1 bounds xy
			const float4 nz = _bvhData[nodeAddr + 2]; // node 0 & 1 bounds z
			float4 tmp = _bvhData[nodeAddr + 3]; // Child indices in x & y

			int2 indices = make_int2(__float_as_int(tmp.x), __float_as_int(tmp.y));

			if(indices.y == 0x80000000) {
				nodeAddr = *(int*)stackPtr;
				leafAddr = indices.x;
				stackPtr -= 4;
				break;
			}

			const float c0lox = n0xy.x * invDir.x - od.x;
			const float c0hix = n0xy.y * invDir.x - od.x;
			const float c0loy = n0xy.z * invDir.y - od.y;
			const float c0hiy = n0xy.w * invDir.y - od.y;
			const float c0loz = nz.x   * invDir.z - od.z;
			const float c0hiz = nz.y   * invDir.z - od.z;
			const float c1loz = nz.z   * invDir.z - od.z;
			const float c1hiz = nz.w   * invDir.z - od.z;
			const float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, 0);
			const float c0max = spanEndKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, 1e20);
			const float c1lox = n1xy.x * invDir.x - od.x;
			const float c1hix = n1xy.y * invDir.x - od.x;
			const float c1loy = n1xy.z * invDir.y - od.y;
			const float c1hiy = n1xy.w * invDir.y - od.y;
			const float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, 0);
			const float c1max = spanEndKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, 1e20);

			bool swp = (c1min < c0min);
			bool traverseChild0 = (c0max >= c0min);
			bool traverseChild1 = (c1max >= c1min);

			if(!traverseChild0 && !traverseChild1)
			{
				nodeAddr = *(int*)stackPtr;
				stackPtr -= 4;
			}
			else
			{
				nodeAddr = (traverseChild0) ? indices.x : indices.y;
				if(traverseChild0 && traverseChild1)
				{
					if(swp)
						swap(nodeAddr, indices.y);
					stackPtr += 4;
					*(int*)stackPtr = indices.y;
				}
			}

			if(nodeAddr < 0 && leafAddr >= 0) // Postpone max 1
			{
				leafAddr = nodeAddr;

				nodeAddr = *(int*)stackPtr;
				stackPtr -= 4;
			}

			unsigned int mask;
			asm("{\n"
				"   .reg .pred p;               \n"
				"setp.ge.s32        p, %1, 0;   \n"
				"vote.ballot.b32    %0,p;       \n"
				"}"
				: "=r"(mask)
				: "r"(leafAddr));

//			int mask = leafAddr >= 0;
			if(!mask)
				break;
		}
		while(leafAddr < 0)
		{
			for(int triAddr = ~leafAddr;; triAddr += 3)
			{
//				float4 vert0 = tex1Dfetch(t_vertices, triAddr);
				float4 vert0 = _vertices[triAddr];
				// Did we reach the terminating point of the triangle(s) in the leaf
				if(__float_as_int(vert0.x) == 0x80000000)
					break;

//				float4 vert1 = tex1Dfetch(t_vertices, triAddr + 1);
//				float4 vert2 = tex1Dfetch(t_vertices, triAddr + 2);
				float4 vert1 = _vertices[triAddr + 1];
				float4 vert2 = _vertices[triAddr + 2];

				float dist = intersectTriangle(vert0, vert1, vert2, _ray);
				if(dist != 0.0f && dist < t)
				{
					t = dist;
					_hitData->m_hitPoint = _ray->m_origin + _ray->m_dir * t;
//					_hitData->m_normal = tex1Dfetch(t_normals, triAddr);
					_hitData->m_normal = _normals[triAddr];
					_hitData->m_color = make_float4(1.f, 1.f, 1.f, 0.0f);
					_hitData->m_emission = make_float4(0.f, 0.0f, 0.0f, 0.0f);
				}
			}
			leafAddr = nodeAddr;
			if(nodeAddr < 0)
			{
				nodeAddr = *(int*)stackPtr;
				stackPtr -= 4;
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
__device__ float4 trace(const Ray *_camray, float4 *_vertices, float4 *_normals, float4 *_bvhData, float4 *_hdr, unsigned int *_seed0, unsigned int *_seed1)
{
	Ray ray = *_camray;

	float4 accum_color = make_float4(0.0f, 0.0f, 0.0f, 0.f);
	float4 mask = make_float4(1.0f, 1.0f, 1.0f, 0.f);

	for(unsigned int bounces = 0; bounces < 5; bounces++)
	{
		vHitData hitData;

//		if(!intersectScene(&ray, _triangleData, _triIdxList, _bvhLimits, _bvhChildrenOrTriangles, &hitData)) {
		if(!intersectScene(&ray, _vertices, _normals, _bvhData, &hitData))
		{
			// Sample the HDR map, based on:
			// http://blog.hvidtfeldts.net/index.php/2012/10/image-based-lighting/
			float2 longlat = make_float2(atan2f(ray.m_dir.x, ray.m_dir.z), acosf(ray.m_dir.y));
			longlat.x = longlat.x < 0 ? longlat.x + 2.0 * PI : longlat.x;
			longlat.x /= 2.0 * PI;
			longlat.y /= PI;

			int x = longlat.x * kHDRwidth;
			int y = longlat.y * kHDRheight;
			int addr = x + y*kHDRwidth;

			accum_color += (mask * 2.0f * _hdr[addr]);
			return accum_color;
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
__global__ void render(cudaSurfaceObject_t _tex, float4 *_colors, float4 *_hdr, float4 *_vertices, float4 *_normals, float4 *_bvhData, float4 *_cam, float4 *_dir, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time)
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

//		float scale = tan(kFov * 0.5);
//		float imageAspectRatio = _w / (float)_h;
//		float dx = (2 * (x + 0.5) / (float)_w - 1) * imageAspectRatio * scale;
//		float dy = (1 - 2 * (y + 0.5) / (float)_h) * scale;

		float4 cx = make_float4(_w * .5135 / _h, 0.0f, 0.0f, 0.0f); // ray direction offset in x direction
		float4 cy = normalize(cross(cx, camera.m_dir)) * .5135; // ray direction offset in y direction (.5135 is field of view angle)

		for(unsigned int s = 0; s < kSamps; s++) {  // samples per pixel
			// compute primary ray direction
//			float4 d = camera.m_dir + make_float4(dx, dy, -1, 0);
			float4 d = camera.m_dir + cx*((.25 + x) / _w - .5) + cy*((.25 + y) / _h - .5);
			// create primary ray, add incoming radiance to pixelcolor
			Ray newcam(camera.m_origin, normalize(d));
//			_colors[ind] += trace(&newcam, _triangleData, _triIdxList, _bvhLimits, _bvhChildrenOrTriangles, &s1, &s2) * (kInvSamps);
			_colors[ind] += trace(&newcam, _vertices, _normals, _bvhData, _hdr, &s1, &s2) * (kInvSamps);
		}

		float coef = 1.f/_frame;
		unsigned char r = (unsigned char)(powf(clamp(_colors[ind].x*coef, 0.0, 1.0), kInvGamma) * 255);
		unsigned char g = (unsigned char)(powf(clamp(_colors[ind].y*coef, 0.0, 1.0), kInvGamma) * 255);
		unsigned char b = (unsigned char)(powf(clamp(_colors[ind].z*coef, 0.0, 1.0), kInvGamma) * 255);

		uchar4 data = make_uchar4(r, g, b, 0xff);
		surf2Dwrite(data, _tex, x*sizeof(uchar4), y);
	}
}

__global__ void hdrDim()
{
	printf("%d %d\n", kHDRwidth, kHDRheight);
}

// 1. Using union
__device__ int floatAsInt( float fval )
{
		union FloatInt {
				float f;
				int   i;
		};

		FloatInt fi;
		fi.f = fval;
		return fi.i;
}

void cu_runRenderKernel(// Buffers
												cudaSurfaceObject_t _texture, float4 *_hdr, float4 *_vertices, float4 *_normals, float4 *_bvhData, unsigned int *_triIdxList,
												// Buffer sizes for texture initialisation
												unsigned int _vertCount, unsigned int _bvhNodeCount, unsigned int _triIdxCount,
												float4 *_colorArr, float4 *_cam, float4 *_dir,
												unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time)
{
	dim3 dimBlock(16, 16);
	dim3 dimGrid((_w / dimBlock.x),
							 (_h / dimBlock.y));

//	render<<<dimGrid, dimBlock>>>(_texture, _triangleData, _triIdxList, _bvhLimits, _bvhChildrenOrTriangles, _colorArr, _cam, _dir, _w, _h, _frame, _time);
	render<<<dimGrid, dimBlock>>>(_texture, _colorArr, _hdr, _vertices, _normals, _bvhData, _cam, _dir, _w, _h, _frame, _time);
//	hdrDim<<<1, 1>>>();
}

void cu_setHDRDim(const unsigned int &_w, const unsigned int &_h)
{
	cudaMemcpyToSymbol(kHDRwidth, &_w, sizeof(unsigned int));
	cudaMemcpyToSymbol(kHDRheight, &_h, sizeof(unsigned int));
}

void cu_fillFloat4(float4 *d_ptr, float4 _val, unsigned int _size)
{
	thrust::device_ptr<float4> ptr = thrust::device_pointer_cast(d_ptr);
	thrust::fill(ptr, ptr + _size, _val);
}
