#include <stdio.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/random.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/device_vector.h>

#include "PathTracer.cuh"
#include "RayIntersection.cuh"
#include "MathHelpers.cuh"
#include "Utilities.cuh"

__constant__ __device__ bool kHasDiffuseMap = false;
__constant__ __device__ bool kHasNormalMap = false;
__constant__ __device__ bool kHasSpecularMap = false;
__constant__ __device__ uint2 kDiffuseDim;
__constant__ __device__ uint2 kNormalDim;
__constant__ __device__ uint2 kSpecularDim;

__constant__ __device__ float kInvGamma = 1.f/2.2f;
__constant__ __device__ float kInvSamps = 1.f/2.f;
__constant__ __device__ unsigned int kSamps = 2;
__constant__ __device__ unsigned int kHDRwidth = 0;
__constant__ __device__ unsigned int kHDRheight = 0;

texture<float4, 1, cudaReadModeElementType> t_hdr;
texture<float4, 1, cudaReadModeElementType> t_diffuse;
texture<float4, 1, cudaReadModeElementType> t_normal;
texture<float4, 1, cudaReadModeElementType> t_specular;

enum Refl_t { SPEC, DIFF };

typedef struct Sphere {
	float m_r;       // radius
	float4 m_pos;
	float4 m_emission;
	float4 m_col;
	Refl_t m_refl;

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
//	{ 1e5f, { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f, 0.0f } }, //Botm
	{ 3.5f, { 15.f, 0.f, 15.f, 0.0f },						{ 0.0f, 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f, 0.0f }, SPEC }, // small sphere 1
	{ 3.5f, { 25.f, 0.f, 15.f, 0.0f },						{ 0.0f, 0.0f, 0.0f, 0.0f }, { 0.4f, 0.4f, 0.4f, 0.0f }, DIFF } // small sphere 2
//	{ 150.0f, { 50.0f, 300.6f - .77f, 81.6f, 0.0f },	/*{ 2.0f, 1.8f, 1.6f, 0.0f }*/{ 2.8f, 1.8f, 1.6f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } }  // Light
};

__device__ __inline__ void swap(int &_a, int &_b)
{
	int tmp = _a;
	_a = _b;
	_b = tmp;
}

__device__ inline bool intersectScene(const Ray *_ray, float4 *_vertices, float4 *_normals, float4 *_bvhData, float2 *_uvs, vHitData *_hitData)
{
	/* initialise t to a very large number,
	so t will be guaranteed to be smaller
	when a hit with the scene occurs */

	int n = sizeof(spheres)/sizeof(Sphere);
	float inf = 1e20f;
	float t = inf;

//	/* check if the ray intersects each sphere in the scene */
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
			_hitData->m_hitType = (int)sphere.m_refl;
		}
	}

	const int EntrypointSentinel = 0x76543210;
	int startNode = 0;
	int traversalStack[64];
	traversalStack[0] = EntrypointSentinel;

	char* stackPtr;											// Current position in traversal stack.
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
			const float4 n0xy = _bvhData[nodeAddr + 0]; // node 0 bounds xy
			const float4 n1xy = _bvhData[nodeAddr + 1]; // node 1 bounds xy
			const float4 nz = _bvhData[nodeAddr + 2]; // node 0 & 1 bounds z
			float4 tmp = _bvhData[nodeAddr + 3]; // Child indices in x & y

			int2 indices = make_int2(__float_as_int(tmp.x), __float_as_int(tmp.y));

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
				float4 vert0 = _vertices[triAddr];
				// Did we reach the terminating point of the triangle(s) in the leaf
				if(__float_as_int(vert0.x) == 0x80000000)
					break;

				float4 vert1 = _vertices[triAddr + 1];
				float4 vert2 = _vertices[triAddr + 2];

				// Check for an intersection (and barycentric coords for uv mapping)
				float4 intersection = intersectTriangle(vert0, vert1, vert2, _ray);
				if(intersection.x > epsilon && intersection.x < t)
				{
					t = intersection.x;
					_hitData->m_hitPoint = _ray->m_origin + _ray->m_dir * t;
					// UVS: (1 - u - v) * t0 + u * t1 + v * t2
//					_hitData->m_color = (1.f - intersection.y - intersection.z) * make_float4(_uvs[triAddr].x, _uvs[triAddr].y, 0.f, 0.f) +
//															intersection.y * make_float4(_uvs[triAddr + 1].x, _uvs[triAddr + 1].y, 0.f, 0.f) +
//															intersection.z * make_float4(_uvs[triAddr + 2].x, _uvs[triAddr + 2].y, 0.f, 0.f);

					float2 uv = (1.f - intersection.y - intersection.z) * _uvs[triAddr] +
											intersection.y * _uvs[triAddr + 1] +
											intersection.z * _uvs[triAddr + 2];

					if(kHasDiffuseMap)
					{
						int x = kDiffuseDim.x * uv.x;
						int y = kDiffuseDim.y * uv.y;
						int addr = clamp(x + y*kDiffuseDim.x, 0, kDiffuseDim.x*kDiffuseDim.y - 1);
						_hitData->m_color = tex1Dfetch(t_diffuse, addr);
					}
					else
					{
						_hitData->m_color = make_float4(0.9f, 0.3f, 0.f, 0.0f);
					}

					if(kHasNormalMap)
					{
						int x = kNormalDim.x * uv.x;
						int y = kNormalDim.y * uv.y;
						int addr = clamp(x + y*kNormalDim.x, 0, kNormalDim.x*kNormalDim.y - 1);
						// Normal map to normals
						float4 normalCS = normalize(cross(vert0 - vert1, vert0 - vert2)) + _hitData->m_hitPoint;
						float4 tangentCS = cross(make_float4(0.f, 1.f, 0.f, 0.f), normal) + _hitData->m_hitPoint;
						float4 bitangentCS = cross(normal, tangent) + _hitData->m_hitPoint;

//						normal = normalize(fs_in.TBN * normal);
						_hitData->m_normal = normalize(2.f * tex1Dfetch(t_normal, addr) - make_float4(1.f, 1.f, 1.f, 0.f));
					}
					else
					{
						_hitData->m_normal = normalize(cross(vert0 - vert1, vert0 - vert2));
					}

//					_hitData->m_color = tex1Dfetch(t_diffuse, t0);
//					_hitData->m_color = tex2D(t_diffuse, uniformDist(rng), uniformDist(rng));

					_hitData->m_emission = make_float4(0.f, 0.0f, 0.0f, 0.0f);
					_hitData->m_hitType = 1;

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

//__device__ float4 trace(const Ray *_camray, float4 *_triangleData, unsigned int *_triIdxList, float2 *_bvhLimits, unsigned int4 *_bvhChildrenOrTriangles, unsigned int *_seed0, unsigned int *_seed1)
//__device__ float4 trace(curandState* randstate, const Ray *_camray, float4 *_vertices, float4 *_normals, float4 *_bvhData, float4 *_hdr)
__device__ float4 trace(const Ray *_camray, float4 *_vertices, float4 *_normals, float4 *_bvhData, float2 *_uvs, float4 *_hdr, unsigned int *_seed0, unsigned int *_seed1)
{
	Ray ray = *_camray;

	float4 accum_color = make_float4(0.0f, 0.0f, 0.0f, 0.f);
	float4 mask = make_float4(1.0f, 1.0f, 1.0f, 0.f);
	float depth = 1.f;

	for(unsigned int bounces = 0; bounces < 4; bounces++)
	{
		vHitData hitData;

		if(!intersectScene(&ray, _vertices, _normals, _bvhData, _uvs, &hitData))
		{
			// Sample the HDR map, based on:
			// http://blog.hvidtfeldts.net/index.php/2012/10/image-based-lighting/
			float2 longlat = make_float2(atan2f(ray.m_dir.x, ray.m_dir.z), acosf(ray.m_dir.y));
			longlat.x = longlat.x < 0 ? longlat.x + 2.0 * PI : longlat.x;
			longlat.x /= 2.0 * PI;
			longlat.y /= PI;

			int x = longlat.x * kHDRwidth;
			int y = longlat.y * kHDRheight;
			int addr = clamp(x + y*kHDRwidth, 0, kHDRwidth*kHDRheight - 1);

			accum_color += (mask * 2.0f * _hdr[addr]);
			accum_color.w = depth;
			return accum_color;
		}

		if(bounces == 0)
		{
			float4 l = ray.m_origin - hitData.m_hitPoint;
			depth = sqrtf(dot(l, l)) / 150.f;
		}

		/* add the colour and light contributions to the accumulated colour */
		accum_color += mask * hitData.m_emission;

		/* compute the surface normal and flip it if necessary to face the incoming ray */
		float4 normal = dot(hitData.m_normal, ray.m_dir) < 0.0f ? hitData.m_normal : hitData.m_normal * (-1.0f);
		ray.m_origin = hitData.m_hitPoint;

		if(hitData.m_hitType == 0)
		{
			ray.m_dir = ray.m_dir - normal * 2.f * dot(normal, ray.m_dir);
			/* add a very small offset to the hitpoint to prevent self intersection */
			ray.m_origin += normal * 0.05f;
		}
		else if(hitData.m_hitType == 1)
		{
			unsigned int seed = hash(_seed0, _seed1);
			thrust::default_random_engine rng(seed);
			thrust::uniform_real_distribution<float> uniformDist(0, 1);

//			/* compute two random numbers to pick a random point on the hemisphere above the hitpoint*/
//			float rand1 = 2.0f * PI * uniformDist(rng);
//			float rand2 = uniformDist(rng);
//	//		float rand1 = 2.0f * PI * curand_uniform(randstate);
//	//		float rand2 = curand_uniform(randstate);
//			float rand2s = sqrt(rand2);

//			/* create a local orthogonal coordinate frame centered at the hitpoint */
//			float4 w = normal;
//			float4 axis = fabs(w.x) > 0.1f ? make_float4(0.0f, 1.0f, 0.0f, 0.f) : make_float4(1.0f, 0.0f, 0.0f, 0.f);
//			float4 u = normalize(cross(axis, w));
//			float4 v = cross(w, u);

//			/* use the coordinate frame and random numbers to compute the next ray direction */
//			float4 newdir = normalize(u * cos(rand1)*rand2s + v*sin(rand1)*rand2s + w*sqrt(1.0f - rand2));
//			ray.m_dir = newdir;

//			/* the mask colour picks up surface colours at each bounce */
//			mask *= hitData.m_color;

//			/* perform cosine-weighted importance sampling for diffuse surfaces*/
//			mask *= dot(newdir, normal);
//			mask *= 2;

			float rouletteRandomFloat = uniformDist(rng);
			float threshold = 0.05f;
			float4 specularColor = make_float4(1.f, 1.f, 1.f, 0.f);  // hard-coded
			bool reflectFromSurface = (rouletteRandomFloat < threshold); //computeFresnel(make_Vec3f(n.x, n.y, n.z), incident, incidentIOR, transmittedIOR, reflectionDirection, transmissionDirection).reflectionCoefficient);

			float4 newdir;
			float4 w = normal;
			float4 axis = fabs(w.x) > 0.1f ? make_float4(0.0f, 1.0f, 0.0f, 0.f) : make_float4(1.0f, 0.0f, 0.0f, 0.f);

			if (reflectFromSurface)
			{ // calculate perfectly specular reflection

				// Ray reflected from the surface. Trace a ray in the reflection direction.
				// TODO: Use Russian roulette instead of simple multipliers!
				// (Selecting between diffuse sample and no sample (absorption) in this case.)

				mask *= specularColor;
				newdir = normalize(ray.m_dir - normal * 2.f * dot(normal, ray.m_dir));
			}
			else
			{  // calculate perfectly diffuse reflection
				/* compute two random numbers to pick a random point on the hemisphere above the hitpoint*/
				float rand1 = 2.0f * PI * uniformDist(rng);
				float rand2 = uniformDist(rng);
				float rand2s = sqrt(rand2);

				/* create a local orthogonal coordinate frame centered at the hitpoint */
				float4 u = normalize(cross(axis, w));
				float4 v = cross(w, u);

				// compute cosine weighted random ray direction on hemisphere
				newdir = normalize(u*cosf(rand1)*rand2s + v*sinf(rand1)*rand2s + w*sqrtf(1 - rand2));

				// multiply mask with colour of object
				mask *= hitData.m_color;
			}

			// offset origin next path segment to prevent self intersection
			ray.m_origin += normal * 0.001f;  // // scene size dependent
			ray.m_dir = newdir;
		}
	}

	accum_color.w = depth;
	return accum_color;
}

//__global__ void render(cudaSurfaceObject_t _tex, float4 *_triangleData, unsigned int *_triIdxList, float2 *_bvhLimits, unsigned int4 *_bvhChildrenOrTriangles,
//											 float4 *_colors, float4 *_cam, float4 *_dir, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time)
__global__ void render(cudaSurfaceObject_t o_tex, cudaSurfaceObject_t o_depth, float4 *io_colors, float4 *_hdr, float4 *_vertices, float4 *_normals, float4 *_bvhData, float2 *_uvs,
											 vCamera _cam, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < _w && y < _h) {

		unsigned int ind = x + y*_w;
		unsigned int s1 = x * _frame;
		unsigned int s2 = y * _time;

		curandState randState;
		curand_init(hash(&s1, &s2) + threadIdx.x, 0, 0, &randState);

		if(_frame == 1) {
			io_colors[ind] = make_float4(0.f, 0.f, 0.f, 0.f);
		}

		vCamera camera;
		camera.m_origin = _cam.m_origin;
		camera.m_dir = _cam.m_dir;
		camera.m_upV = _cam.m_upV;
		camera.m_rightV = _cam.m_rightV;

		float4 cx = _cam.m_fovScale * _w / _h * camera.m_rightV; // ray direction offset in x direction
		float4 cy = _cam.m_fovScale * camera.m_upV; // ray direction offset in y direction (.5135 is field of view angle)

		for(unsigned int s = 0; s < kSamps; s++) {  // samples per pixel
			// compute primary ray direction
			float4 d = camera.m_dir + cx*((.25 + x) / _w - .5) + cy*((.25 + y) / _h - .5);
			// create primary ray, add incoming radiance to pixelcolor
			Ray newcam(camera.m_origin, normalize(d));

//			_colors[ind] += trace(&randState, &newcam, _vertices, _normals, _bvhData, _hdr) * kInvSamps;

			float4 result = trace(&newcam, _vertices, _normals, _bvhData, _uvs, _hdr, &s1, &s2);

			unsigned char depth = (unsigned char)((1.f - result.w) * 255);
			surf2Dwrite(make_uchar4(depth, depth, depth, 0xff), o_depth, x*sizeof(uchar4), y);

			io_colors[ind] += result * kInvSamps;
		}

		float coef = 1.f/_frame;
		float4 color = clamp(io_colors[ind] * coef, 0.f, 1.f);
		unsigned char r = (unsigned char)(powf(color.x, kInvGamma) * 255);
		unsigned char g = (unsigned char)(powf(color.y, kInvGamma) * 255);
		unsigned char b = (unsigned char)(powf(color.z, kInvGamma) * 255);

		uchar4 data = make_uchar4(r, g, b, 0xff);
		surf2Dwrite(data, o_tex, x*sizeof(uchar4), y);
	}
}

void cu_runRenderKernel(// Buffers
												cudaSurfaceObject_t o_texture, cudaSurfaceObject_t o_depth, float4 *_hdr, float4 *_vertices, float4 *_normals, float4 *_bvhData, float2 *_uvs,
												float4 *io_colorArr, vCamera _cam, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time)
{
	dim3 dimBlock(16, 16);
	dim3 dimGrid((_w / dimBlock.x),
							 (_h / dimBlock.y));

	render<<<dimGrid, dimBlock>>>(o_texture, o_depth, io_colorArr, _hdr, _vertices, _normals, _bvhData, _uvs, _cam, _w, _h, _frame, _time);
}

void cu_bindTexture(const float4 *_deviceTexture, const unsigned int _w, const unsigned int _h, const vTextureType &_type)
{
	bool dummyBool = true;
	bool textureLoaded = false;
	uint2 dim = make_uint2(_w, _h);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

	switch(_type)
	{
		case DIFFUSE:
		{
			cudaMemcpyToSymbol(kDiffuseDim, &dim, sizeof(uint2));
			cudaMemcpyFromSymbol(&textureLoaded, kHasDiffuseMap, sizeof(bool));
			if(!textureLoaded)
			{
				cudaBindTexture(NULL, &t_diffuse, _deviceTexture, &channelDesc, _w * _h * sizeof(float4));
				cudaMemcpyToSymbol(kHasDiffuseMap, &dummyBool, sizeof(bool));
			}
		} break;
		case NORMAL:
		{
			cudaMemcpyToSymbol(kNormalDim, &dim, sizeof(uint2));
			cudaMemcpyFromSymbol(&textureLoaded, kHasNormalMap, sizeof(bool));
			if(!textureLoaded)
			{
				cudaBindTexture(NULL, &t_normal, _deviceTexture, &channelDesc, _w * _h * sizeof(float4));
				cudaMemcpyToSymbol(kHasNormalMap, &dummyBool, sizeof(bool));
			}
		} break;
		case SPECULAR:
		{
			cudaMemcpyToSymbol(kSpecularDim, &dim, sizeof(uint2));
			cudaMemcpyFromSymbol(&textureLoaded, kHasSpecularMap, sizeof(bool));
			if(!textureLoaded)
			{
				cudaBindTexture(NULL, &t_specular, _deviceTexture, &channelDesc, _w * _h * sizeof(float4));
				cudaMemcpyToSymbol(kHasSpecularMap, &dummyBool, sizeof(bool));
			}
		} break;
		default: break;
	}
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

void cu_cleanUp()
{
	bool textureLoaded = false;

	cudaMemcpyFromSymbol(&textureLoaded, kHasDiffuseMap, sizeof(bool));
	if(textureLoaded)
	{
		cudaUnbindTexture(t_diffuse);
	}

	cudaMemcpyFromSymbol(&textureLoaded, kHasNormalMap, sizeof(bool));
	if(textureLoaded)
	{
		cudaUnbindTexture(t_normal);
	}

	cudaMemcpyFromSymbol(&textureLoaded, kHasSpecularMap, sizeof(bool));
	if(textureLoaded)
	{
		cudaUnbindTexture(t_specular);
	}
}
