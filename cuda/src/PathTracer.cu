#include <cuda_runtime.h>
#include <thrust/random.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/device_vector.h>

#include "PathTracer.cuh"
#include "RayIntersection.cuh"
#include "MathHelpers.cuh"
#include "Utilities.cuh"

#define RED_SCALE (1.0/1500.0)
#define GREEN_SCALE (1.15/1500.0)
#define BLUE_SCALE (1.66/1500.0)

__constant__ __device__ bool kHasDiffuseMap = false;
__constant__ __device__ bool kHasNormalMap = false;
__constant__ __device__ bool kHasSpecularMap = false;
__constant__ __device__ bool kHasBRDF = false;
__constant__ __device__ bool kMeshInitialised = false;
__constant__ __device__ bool kUseExampleSphere = false;
__constant__ __device__ bool kViewBRDF = false;
__constant__ __device__ bool kUseCornellBox = false;

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

texture<float, 1, cudaReadModeElementType> t_brdf;

enum Refl_t { SPEC, DIFF, BRDF };

typedef struct Sphere {
	float m_r;       // radius
	float4 m_pos;
	float4 m_emission;
	float4 m_col;
	Refl_t m_refl;

	__device__ float intersect(const Ray *_r) const
	{
		float4 op = m_pos - _r->m_origin;
		float t;
		float eps = 1e-4;
		float b = dot(op, _r->m_dir);
		float det = b*b - dot(op, op) + m_r*m_r;
		if(det < 0)
		{
			return 0;
		}
		else
		{
			det = sqrtf(det);
		}
		return (t = b-det) > eps ? t : ((t = b+det) > eps ? t : 0.0);
	}
} Sphere;

__constant__ Sphere cornellBox[] =
{
	{ 160.f, { 0.f, 160.f + 49.f, 0.f, 0.f }, { 4.0f, 3.6f, 3.2f, 0.f }, { 0.f, 0.f, 0.f, 0.f }, DIFF }, // Light
	{ 1e5f, { 1e5f + 50.f, 0.f, 0.f, 0.f }, { 0.075f, 0.025f, 0.025f, 0.f }, { 0.75f, 0.25f, 0.25f, 0.f }, DIFF }, // Right wall
	{ 1e5f, { -1e5f - 50.f, 0.f, 0.f, 0.f }, { 0.025f, 0.075f, 0.025f, 0.f }, { 0.25f, 0.75f, 0.25f, 0.f }, DIFF }, // Left wall
	{ 1e5f, { 0.f, 0.f, -1e5f - 100.f, 0.f }, { 0.f, 0.f, 0.f, 0.f }, { 1.f, 1.f, 1.f, 0.f }, DIFF }, // Back wall
	{ 1e5f, { 0.f, 1e5f + 50.f, 0.f, 0.f }, { 0.f, 0.f, 0.f, 0.f }, { 1.f, 1.f, 1.f, 0.f }, DIFF }, // Ceiling
	{ 1e5f, { 0.f, -1e5f - 50.f, 0.f, 0.f }, { 0.f, 0.f, 0.f, 0.f }, { 1.f, 1.f, 1.f, 0.f }, DIFF } // Floor
};


__constant__ Sphere spheres[] =
{
	{ 3.5f, { 15.f, 0.f, 15.f, 0.f }, { 0.f, 0.f, 0.f, 0.f }, { 0.f, 0.f, 0.f, 0.f }, SPEC }, // Small mirror sphere
	{ 3.5f, { 25.f, 0.f, 15.f, 0.f }, { 0.f, 0.f, 0.f, 0.f }, { 1.f, 1.f, 1.f, 0.f }, DIFF }, // Small gray sphere
};

__constant__ Sphere exampleSphere = { 10.f, { 0.f, 0.f, 0.f, 0.f },{ 0.f, 0.f, 0.f, 0.f }, { 1.f, 1.f, 1.f, 0.f }, DIFF };

__device__ bool intersectScene(const Ray *_ray,
																			float4 *_vertices,
																			float4 *_normals,
																			float4 *_tangents,
																			float4 *_bvhData,
																			float2 *_uvs,
																			vHitData *_hitData)
{
	float inf = 1e20f;
	float t = inf;

	if(kUseCornellBox)
	{
		unsigned int n = sizeof(cornellBox)/sizeof(Sphere);
		for(unsigned int i = 0; i < n; ++i)
		{
			Sphere sphere = cornellBox[i];
			float dist = sphere.intersect(_ray);
			if(dist != 0.f && dist < t)
			{
				t = dist;
				_hitData->m_hitPoint = _ray->m_origin + _ray->m_dir * t;
				_hitData->m_normal = normalize(_hitData->m_hitPoint - sphere.m_pos);

				_hitData->m_color = sphere.m_col;
				_hitData->m_emission = sphere.m_emission;
				_hitData->m_hitType = (int)sphere.m_refl;
				_hitData->m_specularColor = make_float4(0.f, 0.f, 0.f, 0.f);
			}
		}
	}

	int n = sizeof(spheres)/sizeof(Sphere);
	for(int i = 0; i < n; ++i)
	{
		Sphere sphere = spheres[i];
		float dist = sphere.intersect(_ray);
		if(dist != 0.f && dist < t)
		{
			t = dist;
			_hitData->m_hitPoint = _ray->m_origin + _ray->m_dir * t;
			_hitData->m_normal = normalize(_hitData->m_hitPoint - sphere.m_pos);

			_hitData->m_color = sphere.m_col;
			_hitData->m_emission = sphere.m_emission;
			_hitData->m_hitType = (int)sphere.m_refl;
			_hitData->m_specularColor = make_float4(1.f, 1.f, 1.f, 0.f);

			float r = distance(_hitData->m_hitPoint);
			float theta = acosf(_hitData->m_hitPoint.z / r);
			float phi = atan2f(_hitData->m_hitPoint.y, _hitData->m_hitPoint.x);
			_hitData->m_tangent = make_float4(sinf(theta) * cosf(phi), sinf(theta) * sin(phi), cosf(theta), 0.f);
		}
	}

	if(kUseExampleSphere)
	{
		float dist = exampleSphere.intersect(_ray);
		if(dist != 0.f && dist < t)
		{
			t = dist;
			_hitData->m_hitPoint = _ray->m_origin + _ray->m_dir * t;

			// Calculate the uv coordinates (not used for now)
			float u = atan2f(_hitData->m_normal.x, _hitData
											 ->m_normal.z) / (2.f * PI) + 0.5f;
			float v = _hitData->m_normal.y * 0.5f + 0.5f;

			if(kHasDiffuseMap && !kViewBRDF)
			{
				int x = kDiffuseDim.x * u;
				int y = kDiffuseDim.y * v;
				int addr = clamp(x + y*kDiffuseDim.x, 0, kDiffuseDim.x*kDiffuseDim.y - 1);
				_hitData->m_color = tex1Dfetch(t_diffuse, addr);
			}
			else
			{
				_hitData->m_color = exampleSphere.m_col;
			}

			if(kHasNormalMap)
			{
				int x = kNormalDim.x * u;
				int y = kNormalDim.y * v;
				int addr = clamp(x + y*kNormalDim.x, 0, kNormalDim.x*kNormalDim.y - 1);
				// Normal map to normals
				float4 normal = normalize(_hitData->m_hitPoint - exampleSphere.m_pos);
				normal.w = 0.f;

				float r = distance(_hitData->m_hitPoint);
				float theta = acosf(_hitData->m_hitPoint.z / r);
				float phi = atan2f(_hitData->m_hitPoint.y, _hitData->m_hitPoint.x);
				_hitData->m_tangent = make_float4(sinf(theta) * cosf(phi), sinf(theta) * sin(phi), cosf(theta), 0.f);

				float4 bitangent = cross(normal, _hitData->m_tangent);

				mat4 tbn = mat4(_hitData->m_tangent, bitangent, normal);

				float4 normalMap = normalize(2.f * tex1Dfetch(t_normal, addr) - make_float4(1.f, 1.f, 1.f, 0.f));
				_hitData->m_normal = normalize(tbn * normalMap);
			}
			else
			{
				_hitData->m_normal = normalize(_hitData->m_hitPoint - exampleSphere.m_pos);
			}

			if(kHasSpecularMap && !kViewBRDF)
			{
				int x = kSpecularDim.x * u;
				int y = kSpecularDim.y * v;
				int addr = clamp(x + y*kSpecularDim.x, 0, kSpecularDim.x*kSpecularDim.y - 1);
				// Normal map to normals
				_hitData->m_specularColor = tex1Dfetch(t_specular, addr);
			}
			else
			{
				_hitData->m_specularColor = make_float4(0.f, 0.f, 0.f, 0.f);
			}

			_hitData->m_emission = exampleSphere.m_emission;
			_hitData->m_hitType = (int)(kViewBRDF ? BRDF : DIFF);
		}
	}
	else if(kMeshInitialised)
	{
		const int EntrypointSentinel = 0x76543210;
		int startNode = 0;
		int traversalStack[64];
		traversalStack[0] = EntrypointSentinel;

		char* stackPtr;											// Current position in traversal stack.
		int leafAddr;                       // First postponed leaf, non-negative if none.
		int nodeAddr = EntrypointSentinel;  // Non-negative: current internal node, negative: second postponed leaf.
		stackPtr = (char*)&traversalStack[0];
		leafAddr = 0;												// No postponed leaf.
		nodeAddr = startNode;								// Start from the root.


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
						float2 uv = (1.f - intersection.y - intersection.z) * _uvs[triAddr] +
												intersection.y * _uvs[triAddr + 1] +
												intersection.z * _uvs[triAddr + 2];

						float4 tangent =  normalize((1.f - intersection.y - intersection.z) * _tangents[triAddr] +
																				intersection.y * _tangents[triAddr + 1] +
																				intersection.z * _tangents[triAddr + 2]);
						tangent.w = 0.f;

						if(kHasDiffuseMap && !kViewBRDF)
						{
							int x = kDiffuseDim.x * uv.x;
							int y = kDiffuseDim.y * uv.y;
							int addr = clamp(x + y*kDiffuseDim.x, 0, kDiffuseDim.x*kDiffuseDim.y - 1);
							_hitData->m_color = tex1Dfetch(t_diffuse, addr);
						}
						else
						{
							_hitData->m_color = make_float4(1.f, 1.f, 1.f, 0.f);
						}

						// Note that we need tangents to calculate the normals from the normal map
						if(kHasNormalMap && distanceSquared(tangent) > epsilon)
						{
							int x = kNormalDim.x * uv.x;
							int y = kNormalDim.y * uv.y;
							int addr = clamp(x + y*kNormalDim.x, 0, kNormalDim.x*kNormalDim.y - 1);
							// Normal map to normals
							float4 normal = normalize((1.f - intersection.y - intersection.z) * _normals[triAddr] +
																				intersection.y * _normals[triAddr + 1] +
																				intersection.z * _normals[triAddr + 2]);
							normal.w = 0.f;

							float4 bitangent = cross(normal, tangent);

							mat4 tbn = mat4(tangent, bitangent, normal);

							float4 normalMap = normalize(2.f * tex1Dfetch(t_normal, addr) - make_float4(1.f, 1.f, 1.f, 0.f));
							_hitData->m_normal = normalize(tbn * normalMap);
						}
						else
						{
							// Calculate face normal for flat shading
							_hitData->m_normal = normalize(cross(vert0 - vert1, vert0 - vert2));
						}


						if(kHasSpecularMap && !kViewBRDF)
						{
							int x = kSpecularDim.x * uv.x;
							int y = kSpecularDim.y * uv.y;
							int addr = clamp(x + y*kSpecularDim.x, 0, kSpecularDim.x*kSpecularDim.y - 1);
							// Normal map to normals
							_hitData->m_specularColor = tex1Dfetch(t_specular, addr);
						}
						else
						{
							_hitData->m_specularColor = make_float4(0.f, 0.f, 0.f, 0.f);
						}

						_hitData->m_tangent = tangent;
						_hitData->m_emission = make_float4(0.f, 0.f, 0.f, 0.f);
						_hitData->m_hitType = (int)(kViewBRDF ? BRDF : DIFF);
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
	}

	// Returns true if an intersection was found
	return t < inf;
}

inline __device__ int phi_diff_index(float phi_diff)
{
	// Lookup phi_diff index

	// Because of reciprocity, the BRDF is unchanged under
	// phi_diff -> phi_diff + M_PI
	if (phi_diff < 0.0)
			phi_diff += M_PI;

	// In: phi_diff in [0 .. pi]
	// Out: tmp in [0 .. 179]
	return clamp(int(phi_diff * (1.0/PI * (BRDF_SAMPLING_RES_PHI_D / 2))), 0, BRDF_SAMPLING_RES_PHI_D / 2 - 1);
}

inline __device__ int theta_half_index(float theta_half)
{

	// Lookup theta_half index
	// This is a non-linear mapping!
	// In:  [0 .. pi/2]
	// Out: [0 .. 89]
	if (theta_half <= 0.0)
			return 0;

	return clamp(int(sqrtf(theta_half * (2.0/PI)) * BRDF_SAMPLING_RES_THETA_H), 0, BRDF_SAMPLING_RES_THETA_H-1);
}

inline __device__ int theta_diff_index(float theta_diff)
{
	// Lookup theta_diff index
	// In:  [0 .. pi/2]
	// Out: [0 .. 89]
	return clamp(int(theta_diff * (2.0/PI * BRDF_SAMPLING_RES_THETA_D)), 0, BRDF_SAMPLING_RES_THETA_D - 1);
}

__device__ float4 lookupBRDF(const float4 _reflectedDir, const float4 _currentDir, const float4 _normal, const float4 _tangent)
{
	float4 bitangent = cross(_normal, _tangent);

	float4 H = normalize(_reflectedDir - _currentDir);
	float theta_H = acosf(clamp(dot(_normal, H), 0.f, 1.f));
	float theta_diff = acosf(clamp(dot(H, _reflectedDir), 0.f, 1.f));
	float phi_diff = 0.f;

	if (theta_diff < 1e-3)
	{
		// phi_diff indeterminate, use phi_half instead
		phi_diff = atan2f(clamp(-dot(_reflectedDir, bitangent), -1.f, 1.f), clamp(dot(_reflectedDir, _tangent), -1.f, 1.f));
	}
	else if (theta_H > 1e-3)
	{
		// use Gram-Schmidt orthonormalization to find diff basis vectors
		float4 u = -1.f * normalize(_normal - dot(_normal, H) * H);
		float4 v = cross(H, u);
		phi_diff = atan2f(clamp(dot(_reflectedDir, v), -1.f, 1.f), clamp(dot(_reflectedDir, u), -1.f, 1.f));
	}
	else
	{
		theta_H = 0.f;
	}

	// Find index.
	// Note that phi_half is ignored, since isotropic BRDFs are assumed
	int ind = phi_diff_index(phi_diff) +
			theta_diff_index(theta_diff) * BRDF_SAMPLING_RES_PHI_D / 2 +
			theta_half_index(theta_H) * BRDF_SAMPLING_RES_PHI_D / 2 *
			BRDF_SAMPLING_RES_THETA_D;

	int redIndex = ind;
	int greenIndex = ind + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D/2;
	int blueIndex = ind + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D;

	return make_float4(
					tex1Dfetch(t_brdf, redIndex) * RED_SCALE,
					tex1Dfetch(t_brdf, greenIndex) * GREEN_SCALE,
					tex1Dfetch(t_brdf, blueIndex) * BLUE_SCALE,
					0.f);
}

__device__ static unsigned int hash(unsigned int *seed0, unsigned int *seed1)
{
 *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16); // hash the seeds using bitwise AND and bitshifts
 *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

	return *seed0**seed1;
}

__device__ float4 trace(const Ray *_camray,
												float4 *_vertices,
												float4 *_normals,
												float4 *_tangents,
												float4 *_bvhData,
												float2 *_uvs,
												float4 *_hdr,
												float _fresnelCoef,
												float _fresnelPow,
												unsigned int *_seed0,
												unsigned int *_seed1)
{
	Ray ray = *_camray;

	float4 accum_color = make_float4(0.0f, 0.f, 0.f, 0.f);
	float4 mask = make_float4(1.0f, 1.0f, 1.0f, 0.f);
	float depth = 1.f;

	for(unsigned int bounces = 0; bounces < 4; bounces++)
	{
		vHitData hitData;

		if(!intersectScene(&ray, _vertices, _normals, _tangents, _bvhData, _uvs, &hitData))
		{
			if(!kUseCornellBox)
			{
				// Sample the HDRI map, based on:
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
			else
			{
				return make_float4(0.f, 0.f, 0.f, 0.f);
			}
		}

		if(bounces == 0)
		{
			// Render depth
			float4 l = ray.m_origin - hitData.m_hitPoint;
			depth = sqrtf(dot(l, l)) / 150.f;
		}

		// Add the colour and light contributions to the accumulated colour
		accum_color += mask * hitData.m_emission;

		// Next ray's origin is at the hitpoint
		ray.m_origin = hitData.m_hitPoint;
		float4 normal = hitData.m_normal;

		unsigned int seed = hash(_seed0, _seed1);
		thrust::default_random_engine rng(seed);
		thrust::uniform_real_distribution<float> uniformDist(0, 1);

		if(hitData.m_hitType == 0)
		{
			ray.m_dir = ray.m_dir - normal * 2.f * dot(normal, ray.m_dir);
			ray.m_origin += normal * 0.05f;
		}
		else if(hitData.m_hitType == 1)
		{
			float angleOfIncidence = dot(hitData.m_normal, -1.f * ray.m_dir);
			float fresnelEffect = lerp(powf(1.f - angleOfIncidence, _fresnelPow), 1.f, _fresnelCoef) * hitData.m_specularColor.x;

			bool reflectFromSurface = (uniformDist(rng) < fresnelEffect);

			float4 newdir;
			float4 w = normal;
			float4 axis = fabs(w.x) > 0.1f ? make_float4(0.0f, 1.0f, 0.f, 0.f) : make_float4(1.0f, 0.f, 0.f, 0.f);

			if(reflectFromSurface)
			{
				mask *= hitData.m_specularColor;
				newdir = normalize(ray.m_dir - normal * 2.f * dot(normal, ray.m_dir));
			}
			else
			{
				// compute two random numbers to pick a random point on the hemisphere above the hitpoint
				float rand1 = 2.0f * PI * uniformDist(rng);
				float rand2 = uniformDist(rng);
				float rand2s = sqrt(rand2);

				// create a local orthogonal coordinate frame centered at the hitpoint
				float4 u = normalize(cross(axis, w));
				float4 v = cross(w, u);

				// compute cosine weighted random ray direction on hemisphere
				newdir = normalize(u*cosf(rand1)*rand2s + v*sinf(rand1)*rand2s + w*sqrtf(1 - rand2));

				// multiply mask with colour of object
				mask *= hitData.m_color;
				mask *= dot(newdir, normal);
				mask *= 2.f;
			}

			ray.m_origin += normal * 0.05f;
			ray.m_dir = newdir;
		}
		else if(hitData.m_hitType == 2)
		{
			float4 newdir;
			float4 w = normal;
			float4 axis = fabs(w.x) > 0.1f ? make_float4(0.0f, 1.0f, 0.f, 0.f) : make_float4(1.0f, 0.f, 0.f, 0.f);

			// compute two random numbers to pick a random point on the hemisphere above the hitpoint
			float rand1 = 2.0f * PI * uniformDist(rng);
			float rand2 = uniformDist(rng);
			float rand2s = sqrt(rand2);

			/* create a local orthogonal coordinate frame centered at the hitpoint */
			float4 u = normalize(cross(axis, w));
			float4 v = cross(w, u);

			// compute cosine weighted random ray direction on hemisphere
			newdir = normalize(u*cosf(rand1)*rand2s + v*sinf(rand1)*rand2s + w*sqrtf(1 - rand2));

			if(kHasBRDF)
			{
				float dw = 24 * powf(newdir.x*newdir.x +
														newdir.y*newdir.y +
														newdir.z*newdir.z, -1.5);
				mask *= dw * max(lookupBRDF(newdir, ray.m_dir, hitData.m_normal, hitData.m_tangent), make_float4(0.f, 0.f, 0.f, 0.f));
			}
			else
			{
				// multiply mask with colour of object
				mask *= hitData.m_color;
				mask *= dot(newdir, normal);
				mask *= 2.f;
			}

			ray.m_origin += normal * 0.05f;
			ray.m_dir = newdir;
		}
	}

	accum_color.w = depth;
	return accum_color;
}

__global__ void render(cudaSurfaceObject_t o_tex,
											 cudaSurfaceObject_t o_depth,
											 float4 *io_colors,
											 float4 *_hdr,
											 float4 *_vertices,
											 float4 *_normals,
											 float4 *_tangents,
											 float4 *_bvhData,
											 float2 *_uvs,
											 vCamera _cam,
											 unsigned int _w,
											 unsigned int _h,
											 unsigned int _frame,
											 unsigned int _time,
											 float _fresnelCoef,
											 float _fresnelPow)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < _w && y < _h) {

		unsigned int ind = x + y*_w;
		unsigned int s1 = x * _frame;
		unsigned int s2 = y * _time;

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

			float4 result = trace(&newcam, _vertices, _normals, _tangents, _bvhData, _uvs, _hdr, _fresnelCoef, _fresnelPow, &s1, &s2);

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

void cu_runRenderKernel(cudaSurfaceObject_t o_texture,
												cudaSurfaceObject_t o_depth,
												float4 *_hdr,
												float4 *_vertices,
												float4 *_normals,
												float4 *_tangents,
												float4 *_bvhData,
												float2 *_uvs,
												float4 *io_colorArr,
												vCamera _cam,
												unsigned int _w,
												unsigned int _h,
												unsigned int _frame,
												unsigned int _time,
												float _fresnelCoef,
												float _fresnelPow)
{
	dim3 dimBlock(16, 16);
	dim3 dimGrid((_w / dimBlock.x),
							 (_h / dimBlock.y));

	render<<<dimGrid, dimBlock>>>(o_texture, o_depth, io_colorArr, _hdr, _vertices, _normals, _tangents, _bvhData, _uvs, _cam, _w, _h, _frame, _time, _fresnelCoef, _fresnelPow);
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
			cudaMemcpyFromSymbol(&textureLoaded, kHasDiffuseMap, sizeof(bool));
			if(textureLoaded)
			{
				cudaUnbindTexture(t_diffuse);
			}
			else
			{
				cudaMemcpyToSymbol(kHasDiffuseMap, &dummyBool, sizeof(bool));
			}

			cudaMemcpyToSymbol(kDiffuseDim, &dim, sizeof(uint2));
			cudaBindTexture(NULL, &t_diffuse, _deviceTexture, &channelDesc, _w * _h * sizeof(float4));
		} break;
		case NORMAL:
		{
			cudaMemcpyFromSymbol(&textureLoaded, kHasNormalMap, sizeof(bool));
			if(textureLoaded)
			{
				cudaUnbindTexture(t_normal);
			}
			else
			{
				cudaMemcpyToSymbol(kHasNormalMap, &dummyBool, sizeof(bool));
			}

			cudaMemcpyToSymbol(kNormalDim, &dim, sizeof(uint2));
			cudaBindTexture(NULL, &t_normal, _deviceTexture, &channelDesc, _w * _h * sizeof(float4));
		} break;
		case SPECULAR:
		{
			cudaMemcpyFromSymbol(&textureLoaded, kHasSpecularMap, sizeof(bool));
			if(textureLoaded)
			{
				cudaUnbindTexture(t_specular);
			}
			else
			{
				cudaMemcpyToSymbol(kHasSpecularMap, &dummyBool, sizeof(bool));
			}

			cudaMemcpyToSymbol(kSpecularDim, &dim, sizeof(uint2));
			cudaBindTexture(NULL, &t_specular, _deviceTexture, &channelDesc, _w * _h * sizeof(float4));
		} break;
		default: break;
	}
}

void cu_bindBRDF(const float *_brdf)
{
	bool dummyBool = true;
	bool brdfLoaded = false;
	uint dim = BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D * BRDF_SAMPLING_RES_PHI_D / 2;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	cudaMemcpyFromSymbol(&brdfLoaded, kHasBRDF, sizeof(bool));
	if(brdfLoaded)
	{
		cudaUnbindTexture(t_brdf);
	}
	else
	{
		cudaMemcpyToSymbol(kHasBRDF, &dummyBool, sizeof(bool));
	}

	cudaBindTexture(NULL, &t_brdf, _brdf, &channelDesc, dim * 3 * sizeof(float));
}

void cu_useExampleSphere(const bool &_newVal)
{
	cudaMemcpyToSymbol(kUseExampleSphere, &_newVal, sizeof(bool));
}

void cu_useBRDF(const bool &_newVal)
{
	cudaMemcpyToSymbol(kViewBRDF, &_newVal, sizeof(bool));
}

void cu_useCornellBox(const bool &_newVal)
{
	cudaMemcpyToSymbol(kUseCornellBox, &_newVal, sizeof(bool));
}

void cu_setHDRDim(const unsigned int &_w, const unsigned int &_h)
{
	cudaMemcpyToSymbol(kHDRwidth, &_w, sizeof(unsigned int));
	cudaMemcpyToSymbol(kHDRheight, &_h, sizeof(unsigned int));
}

void cu_meshInitialised()
{
	bool dummyBool = true;
	cudaMemcpyToSymbol(kMeshInitialised, &dummyBool, sizeof(bool));
}

void cu_fillFloat4(float4 *_dPtr, const float4 _val, const unsigned int _size)
{
	thrust::device_ptr<float4> ptr = thrust::device_pointer_cast(_dPtr);
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
