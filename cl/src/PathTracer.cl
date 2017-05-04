///
/// \file PathTracer.cl
/// \brief Device (OpenCL) implementation of the path tracer. As the implementation is pretty much the same as
///				 the Cuda one. The documentation for most everything can be found under the cuda docs.
///

#include "cl/include/PathTracer.h"
#include "cl/include/RayIntersection.h"
#include "cl/include/Utilities.h"

#define BRDF_SAMPLING_RES_THETA_H       90
#define BRDF_SAMPLING_RES_THETA_D       90
#define BRDF_SAMPLING_RES_PHI_D         360

#define RED_SCALE (1.0/1500.0)
#define GREEN_SCALE (1.15/1500.0)
#define BLUE_SCALE (1.66/1500.0)

__constant float invGamma = 1.f/2.2f;
__constant float invSamps = 1.f/2.f;
__constant unsigned int samps = 2;

__constant Sphere spheres[] =
{
	{ 3.5f, { 15.f, 0.f, 15.f, 0.f }, { 0.f, 0.f, 0.f, 0.f }, { 0.f, 0.f, 0.f, 0.f }, SPEC }, // Small mirror sphere
	{ 3.5f, { 25.f, 0.f, 15.f, 0.f }, { 0.f, 0.f, 0.f, 0.f }, { 1.f, 1.f, 1.f, 0.f }, DIFF }, // Small gray sphere
};

__constant Sphere cornellBox[] =
{
	{ 160.f, { 0.f, 160.f + 49.f, 0.f, 0.f }, { 4.f, 3.6f, 3.2f, 0.f }, { 0.f, 0.f, 0.f, 0.f }, DIFF }, // Light
	{ 1e5f, { 1e5f + 50.f, 0.f, 0.f, 0.f }, { 0.075f, 0.025f, 0.025f, 0.f }, { 0.75f, 0.25f, 0.25f, 0.f }, DIFF }, // Right wall
	{ 1e5f, { -1e5f - 50.f, 0.f, 0.f, 0.f }, { 0.025f, 0.075f, 0.025f, 0.f }, { 0.25f, 0.75f, 0.25f, 0.f }, DIFF }, // Left wall
	{ 1e5f, { 0.f, 0.f, -1e5f - 100.f, 0.f }, { 0.f, 0.f, 0.f, 0.f }, { 1.f, 1.f, 1.f, 0.f }, DIFF }, // Back wall
	{ 1e5f, { 0.f, 1e5f + 50.f, 0.f, 0.f }, { 0.f, 0.f, 0.f, 0.f }, { 1.f, 1.f, 1.f, 0.f }, DIFF }, // Ceiling
	{ 1e5f, { 0.f, -1e5f - 50.f, 0.f, 0.f }, { 0.f, 0.f, 0.f, 0.f }, { 1.f, 1.f, 1.f, 0.f }, DIFF } // Floor
};

__constant Sphere exampleSphere = { 10.f, { 0.f, 0.f, 0.f, 0.f },{ 0.f, 0.f, 0.f, 0.f }, { 1.f, 1.f, 1.f, 0.f }, DIFF };

Ray createRay(float4 _o, float4 _d)
{
	Ray ray;
	ray.m_origin = _o;
  ray.m_dir = _d;
	return ray;
}

unsigned int floatAsInt(const float _a)
{
  union
  {
    float a;
    unsigned int b;
  } c;
  c.a = _a;

  return c.b;
}

static float get_random(unsigned int *io_seed0, unsigned int *io_seed1)
{
  /* hash the seeds using bitwise AND operations and bitshifts */
	*io_seed0 = 36969 * ((*io_seed0) & 65535) + ((*io_seed0) >> 16);
	*io_seed1 = 18000 * ((*io_seed1) & 65535) + ((*io_seed1) >> 16);

	unsigned int ires = ((*io_seed0) << 16) + (*io_seed1);

  /* use union struct to convert int to float */
  union {
    float f;
    unsigned int ui;
  } res;

  res.ui = (ires & 0x007fffff) | 0x40000000;  /* bitwise AND, bitwise OR */
	return (res.f - 2.f) / 2.f;
}

bool intersectScene(const Ray *_ray,
                    __global const float4 *_vertices,
                    __global const float4 *_normals,
                    __global const float4 *_tangents,
										__global const float4 *normalMapvhNodes,
                    __global const float2 *_uvs,
                    __read_only image2d_t _diffuse,
                    __read_only image2d_t _normal,
                    __read_only image2d_t _specular,
                    bool _hasDiffuseMap,
                    bool _hasNormalMap,
                    bool _hasSpecularMap,
										bool _useCornellBox,
										bool _useExampleSphere,
										bool _meshInitialised,
										bool _viewBRDF,
                    vHitData *_hitData)
{
	float inf = 1e20f;
	float t = inf;

	if(_useCornellBox)
	{
		unsigned int n = sizeof(cornellBox)/sizeof(Sphere);
		for(unsigned int i = 0; i < n; ++i)
		{
			Sphere sphere = cornellBox[i];
			float dist = intersectSphere(&sphere, _ray);
			if(dist != 0.f && dist < t)
			{
				t = dist;
				_hitData->m_hitPoint = _ray->m_origin + _ray->m_dir * t;
				_hitData->m_normal = normalize(_hitData->m_hitPoint - sphere.m_pos);

				_hitData->m_color = sphere.m_col;
				_hitData->m_emission = sphere.m_emission;
				_hitData->m_hitType = (int)sphere.m_refl;
				_hitData->m_specularColor = (float4)(0.f, 0.f, 0.f, 0.f);
			}
		}
	}

	unsigned int n = sizeof(spheres)/sizeof(Sphere);
	for(int i = 0; i < n; i++)
	{
		Sphere sphere = spheres[i];
		float dist = intersectSphere(&sphere, _ray);
		if(dist != 0.f && dist < t)
		{
			t = dist;
			_hitData->m_hitPoint = _ray->m_origin + _ray->m_dir * t;
			_hitData->m_normal = normalize(_hitData->m_hitPoint - sphere.m_pos);

			_hitData->m_color = sphere.m_col;
			_hitData->m_emission = sphere.m_emission;
			_hitData->m_hitType = (int)sphere.m_refl;
			_hitData->m_specularColor = (float4)(1.f, 1.f, 1.f, 0.f);
		}
  }

	if(_useExampleSphere)
	{
		Sphere sphere = exampleSphere;
		float dist = intersectSphere(&sphere, _ray);
		if(dist != 0.f && dist < t)
		{
			t = dist;
			_hitData->m_hitPoint = _ray->m_origin + _ray->m_dir * t;

			// Calculate the uv coordinates (not used for now)
			float u = atan2(_hitData->m_normal.x, _hitData->m_normal.z) / (2.f * PI) + 0.5f;
			float v = _hitData->m_normal.y * 0.5f + 0.5f;

			if(_hasDiffuseMap && !_viewBRDF)
			{
				int x = get_image_width(_diffuse) * u;
				int y = get_image_width(_diffuse) * v;
				_hitData->m_color = read_imagef(_diffuse, (int2)(x, y));
			}
			else
			{
				_hitData->m_color = exampleSphere.m_col;
			}

//			if(_hasNormalMap)
//			{
//				int x = get_image_width(_diffuse) * u;
//				int y = get_image_width(_diffuse) * v;
//				// Normal map to normals
//				float4 normal = normalize(_hitData->m_hitPoint - exampleSphere.m_pos);
//				normal.w = 0.f;

//				float r = distance(_hitData->m_hitPoint);
//				float theta = acosf(_hitData->m_hitPoint.z / r);
//				float phi = atan2f(_hitData->m_hitPoint.y, _hitData->m_hitPoint.x);
//				_hitData->m_tangent = (float4)(sinf(theta) * cosf(phi), sinf(theta) * sin(phi), cosf(theta), 0.f);

//				float4 bitangent = cross(normal, _hitData->m_tangent);

//				float4 normalMap = normalize(2.f * read_imagef(_normal, (int2)(x, y)) - (float4)(1.f, 1.f, 1.f, 0.f));

//				// Matrix multiplication TBN (tangent, bitangent, normal) * normal map
//				float4 worldSpaceNormal = (float4)(tangent.x * normalMap.x + bitangent.x * normalMap.y + normal.x * normalMap.z,
//																				 tangent.y * normalMap.x + bitangent.y * normalMap.y + normal.y * normalMap.z,
//																				 tangent.z * normalMap.x + bitangent.z * normalMap.y + normal.z * normalMap.z,
//																				 tangent.w * normalMap.x + bitangent.w * normalMap.y + normal.w * normalMap.z + 1.f * normalMap.w);
//				_hitData->m_normal = normalize(worldSpaceNormal);
//			}
//			else
//			{
				_hitData->m_normal = normalize(_hitData->m_hitPoint - exampleSphere.m_pos);
//			}

			if(_hasSpecularMap && !_viewBRDF)
			{
				int x = get_image_width(_specular) * u;
				int y = get_image_width(_specular) * v;
				// Normal map to normals
				_hitData->m_specularColor = read_imagef(_specular, (int2)(u, v));
			}
			else
			{
				_hitData->m_specularColor = (float4)(0.f, 0.f, 0.f, 0.f);
			}

			_hitData->m_emission = exampleSphere.m_emission;
			_hitData->m_hitType = (int)(_viewBRDF ? BRDF : DIFF);
		}
	}
	else if(_meshInitialised)
	{
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

		while(nodeAddr != EntrypointSentinel)
		{
			while((unsigned int)nodeAddr < (unsigned int)EntrypointSentinel)
			{
				const float4 n0xy = normalMapvhNodes[nodeAddr + 0]; // node 0 bounds xy
				const float4 n1xy = normalMapvhNodes[nodeAddr + 1]; // node 1 bounds xy
				const float4 nz = normalMapvhNodes[nodeAddr + 2]; // node 0 & 1 bounds z
				float4 tmp = normalMapvhNodes[nodeAddr + 3]; // Child indices in x & y

				uint2 indices = (uint2)(floatAsInt(tmp.x), floatAsInt(tmp.y));

				if(indices.y == 0x80000000) {
					nodeAddr = *(int*)stackPtr;
					leafAddr = indices.x;
					stackPtr -= 4;
					break;
				}

				float c0min, c1min, c0max, c1max;
				bool traverseChild0 = intersectCFBVH(_ray, (float3)(n0xy.x, n0xy.z, nz.x), (float3)(n0xy.y, n0xy.w, nz.y), &c0min, &c0max);
				bool traverseChild1 = intersectCFBVH(_ray, (float3)(n1xy.x, n1xy.z, nz.z), (float3)(n1xy.y, n1xy.w, nz.w), &c1min, &c1max);
				bool swp = (c1min < c0min);

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
						if(swp) {
							int tmp = nodeAddr;
							nodeAddr = indices.y;
							indices.y = tmp;
						}
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

				int mask = (leafAddr >= 0);
				if(!mask)
					break;
			}
			while(leafAddr < 0)
			{
				for(int triAddr = ~leafAddr;; triAddr += 3)
				{
					float4 vert0 = _vertices[triAddr];
					// Did we reach the terminating point of the triangle(s) in the leaf
					if(floatAsInt(vert0.x) == 0x80000000)
						break;
					float4 vert1 = _vertices[triAddr + 1];
					float4 vert2 = _vertices[triAddr + 2];

					float4 intersection = intersectTriangle(vert0, vert1, vert2, _ray);
					if(intersection.x != 0.f && intersection.x < t)
					{
						t = intersection.x;
						_hitData->m_hitPoint = _ray->m_origin + _ray->m_dir * t;

						float2 uv = (1.f - intersection.y - intersection.z) * _uvs[triAddr] +
												intersection.y * _uvs[triAddr + 1] +
												intersection.z * _uvs[triAddr + 2];

						float4 tangent = normalize((1.f - intersection.y - intersection.z) * _tangents[triAddr] +
																			 intersection.y * _tangents[triAddr + 1] +
																			 intersection.z * _tangents[triAddr + 2]);
						tangent.w = 0.f;

						if(_hasDiffuseMap && !_viewBRDF)
						{
							int x = get_image_width(_diffuse) * uv.x;
							int y = get_image_height(_diffuse) * uv.y;
							_hitData->m_color = read_imagef(_diffuse, (int2)(x, y));
						}
						else
						{
							_hitData->m_color = (float4)(1.f, 1.f, 1.f, 0.f);
						}

						if(_hasSpecularMap && !_viewBRDF)
						{
							int x = get_image_width(_specular) * uv.x;
							int y = get_image_height(_specular) * uv.y;
							_hitData->m_specularColor = read_imagef(_specular, (int2)(x, y));
						}
						else
						{
							_hitData->m_specularColor = (float4)(0.f, 0.f, 0.f, 0.f);
						}

						if(_hasNormalMap)
						{
							int x = get_image_width(_normal) * uv.x;
							int y = get_image_height(_normal) * uv.y;
							float4 normal = normalize((1.f - intersection.y - intersection.z) * _normals[triAddr] +
																				intersection.y * _normals[triAddr + 1] +
																				intersection.z * _normals[triAddr + 2]);
							normal.w = 0.f;

							float4 bitangent = cross(normal, tangent);
							float4 normalMap = normalize(2.f * read_imagef(_normal, (int2)(x, y)) - (float4)(1.f, 1.f, 1.f, 0.f));

							// Matrix multiplication TBN (tangent, bitangent, normal) * normal map
							float4 worldSpaceNormal = (float4)(tangent.x * normalMap.x + bitangent.x * normalMap.y + normal.x * normalMap.z,
																								 tangent.y * normalMap.x + bitangent.y * normalMap.y + normal.y * normalMap.z,
																								 tangent.z * normalMap.x + bitangent.z * normalMap.y + normal.z * normalMap.z,
																								 tangent.w * normalMap.x + bitangent.w * normalMap.y + normal.w * normalMap.z + 1.f * normalMap.w);

							_hitData->m_normal = normalize(normalMap + normal);
//							_hitData->m_normal = normalize(worldSpaceNormal);
						}
						else
						{
							// Calculate face normal for flat shading
							_hitData->m_normal = normalize(cross(vert0 - vert1, vert0 - vert2));
						}

						_hitData->m_tangent = tangent;
						_hitData->m_emission = (float4)(0.f, 0.f, 0.f, 0.f);
						_hitData->m_hitType = (int)(_viewBRDF ? BRDF : DIFF);
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

	return t < inf;
}

inline int phi_diff_index(float phi_diff)
{
	// Lookup phi_diff index

	// Because of reciprocity, the BRDF is unchanged under
	// phi_diff -> phi_diff + M_PI
	if (phi_diff < 0.0)
			phi_diff += M_PI;

	// In: phi_diff in [0 .. pi]
	// Out: tmp in [0 .. 179]
	return clamp((int)(phi_diff * (1.0/PI * (BRDF_SAMPLING_RES_PHI_D / 2))), 0, BRDF_SAMPLING_RES_PHI_D / 2 - 1);
}

inline int theta_half_index(float theta_half)
{

	// Lookup theta_half index
	// This is a non-linear mapping!
	// In:  [0 .. pi/2]
	// Out: [0 .. 89]
	if (theta_half <= 0.0)
			return 0;

	return clamp((int)(sqrt(theta_half * (2.0/PI)) * BRDF_SAMPLING_RES_THETA_H), 0, BRDF_SAMPLING_RES_THETA_H-1);
}

inline int theta_diff_index(float theta_diff)
{
	// Lookup theta_diff index
	// In:  [0 .. pi/2]
	// Out: [0 .. 89]
	return clamp((int)(theta_diff * (2.0/PI * BRDF_SAMPLING_RES_THETA_D)), 0, BRDF_SAMPLING_RES_THETA_D - 1);
}

float4 lookupBRDF(__global const float *normalMaprdf,
									const float4 _reflectedDir,
									const float4 _currentDir,
									const float4 _normal,
									const float4 _tangent)
{
	float4 bitangent = cross(_normal, _tangent);

	float4 H = normalize(_reflectedDir - _currentDir);
	float theta_H = acos(clamp(dot(_normal, H), 0.f, 1.f));
	float theta_diff = acos(clamp(dot(H, _reflectedDir), 0.f, 1.f));
	float phi_diff = 0.f;

	if (theta_diff < 1e-3)
	{
		// phi_diff indeterminate, use phi_half instead
		phi_diff = atan2(clamp(-dot(_reflectedDir, bitangent), -1.f, 1.f), clamp(dot(_reflectedDir, _tangent), -1.f, 1.f));
	}
	else if (theta_H > 1e-3)
	{
		// use Gram-Schmidt orthonormalization to find diff basis vectors
		float4 u = -1.f * normalize(_normal - dot(_normal, H) * H);
		float4 v = cross(H, u);
		phi_diff = atan2(clamp(dot(_reflectedDir, v), -1.f, 1.f), clamp(dot(_reflectedDir, u), -1.f, 1.f));
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

	return (float4)(normalMaprdf[redIndex] * RED_SCALE, normalMaprdf[greenIndex] * GREEN_SCALE,	normalMaprdf[blueIndex] * BLUE_SCALE, 0.f);
}

float4 trace(const Ray* _camray,
             __global const float4 *_vertices,
             __global const float4 *_normals,
             __global const float4 *_tangents,
						 __global const float4 *normalMapvhNodes,
             __global const float2 *_uvs,
             __read_only image2d_t _hdr,
             __read_only image2d_t _diffuse,
             __read_only image2d_t _normal,
             __read_only image2d_t _specular,
						 float _fresnelPow,
						 float _fresnelCoef,
             bool _hasDiffuseMap,
             bool _hasNormalMap,
             bool _hasSpecularMap,
						 bool _useCornellBox,
						 bool _useExampleSphere,
						 bool _meshInitialised,
						 bool _viewBRDF,
						 bool _hasBRDF,
						 __global const float *normalMaprdf,
             unsigned int *_seed0,
             unsigned int *_seed1)
{
	Ray ray = *_camray;

	float4 accum_color = (float4)(0.f, 0.f, 0.f, 0.f);
	float4 mask = (float4)(1.f, 1.f, 1.f, 0.f);

	for(unsigned int bounces = 0; bounces < 4; ++bounces)
	{
		vHitData hitData;

		/* if ray misses scene, return background colour */
		if(!intersectScene(&ray, _vertices, _normals, _tangents, normalMapvhNodes, _uvs, _diffuse, _normal, _specular, _hasDiffuseMap, _hasNormalMap, _hasSpecularMap, _useCornellBox, _useExampleSphere, _meshInitialised, _viewBRDF, &hitData))
		{
			if(!_useCornellBox)
			{
				// Sample the HDRI map, based on:
				// http://blog.hvidtfeldts.net/index.php/2012/10/image-based-lighting/
				float2 longlat = (float2)(atan2(ray.m_dir.x, ray.m_dir.z), acos(ray.m_dir.y));
				longlat.x = longlat.x < 0 ? longlat.x + 2.0 * PI : longlat.x;
				longlat.x /= 2.0 * PI;
				longlat.y /= PI;

				int2 uv = (int2)(get_image_width(_hdr) * longlat.x, get_image_height(_hdr) * longlat.y);

				accum_color += (mask * 2.f * read_imagef(_hdr, uv));
				return accum_color;
			}
			else
			{
				return (float4)(0.f, 0.f, 0.f, 0.f);
			}
    }

		if(bounces == 0)
		{
			// Render depth
			float4 l = ray.m_origin - hitData.m_hitPoint;
//			depth = sqrt(dot(l, l)) / 150.f;
		}

		// Add the colour and light contributions to the accumulated colour
		accum_color += mask * hitData.m_emission;

		// Next ray's origin is at the hitpoint
		ray.m_origin = hitData.m_hitPoint;
		float4 normal = hitData.m_normal;

		if(hitData.m_hitType == 0)
		{
			ray.m_dir = normalize(ray.m_dir - normal * 2.f * dot(normal, ray.m_dir));
			ray.m_origin += normal * 0.05f;
		}
		else if(hitData.m_hitType == 1)
		{
			float angleOfIncidence = dot(normal, -1.f * ray.m_dir);
			float fresnelEffect = mix(pow(1.f - angleOfIncidence, _fresnelPow), 1.f, _fresnelCoef) * hitData.m_specularColor.x;

			bool reflectFromSurface = (get_random(_seed0, _seed1) < fresnelEffect);

			float4 newdir;
			float4 w = normal;
			float4 axis = fabs(w.x) > 0.1f ? (float4)(0.0f, 1.0f, 0.f, 0.f) : (float4)(1.0f, 0.f, 0.f, 0.f);

			if(reflectFromSurface)
			{
				mask *= hitData.m_specularColor;
				newdir = normalize(ray.m_dir - normal * 2.f * dot(normal, ray.m_dir));
			}
			else
			{
				// compute two random numbers to pick a random point on the hemisphere above the hitpoint
				float rand1 = 2.f * PI * get_random(_seed0, _seed1);
				float rand2 = get_random(_seed0, _seed1);
				float rand2s = sqrt(rand2);

				// create a local orthogonal coordinate frame centered at the hitpoint
				float4 u = normalize(cross(axis, w));
				float4 v = cross(w, u);

				// compute cosine weighted random ray direction on hemisphere
				newdir = normalize(u*cos(rand1)*rand2s + v*sin(rand1)*rand2s + w*sqrt(1 - rand2));

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
			float4 axis = fabs(w.x) > 0.1f ? (float4)(0.0f, 1.0f, 0.f, 0.f) : (float4)(1.0f, 0.f, 0.f, 0.f);

			// compute two random numbers to pick a random point on the hemisphere above the hitpoint
			float rand1 = 2.f * PI * get_random(_seed0, _seed1);
			float rand2 = get_random(_seed0, _seed1);
			float rand2s = sqrt(rand2);

			/* create a local orthogonal coordinate frame centered at the hitpoint */
			float4 u = normalize(cross(axis, w));
			float4 v = cross(w, u);

			// compute cosine weighted random ray direction on hemisphere
			newdir = normalize(u*cos(rand1)*rand2s + v*sin(rand1)*rand2s + w*sqrt(1 - rand2));

			if(_hasBRDF)
			{
				float dw = 24.f * pow(newdir.x*newdir.x +
															newdir.y*newdir.y +
															newdir.z*newdir.z, -1.5f);
				mask *= dw * max(lookupBRDF(normalMaprdf, newdir, ray.m_dir, hitData.m_normal, hitData.m_tangent), (float4)(0.f, 0.f, 0.f, 0.f));
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

	return accum_color;
}

__kernel void render(__write_only image2d_t _texture,
                     __global const float4 *_vertices,
                     __global const float4 *_normals,
                     __global const float4 *_tangents,
										 __global const float4 *normalMapvhNodes,
                     __global const float2 *_uvs,
										 __global const float *normalMaprdf,
                     __global float4 *_colors,
                     __read_only image2d_t _hdr,
                     __read_only image2d_t _diffuse,
                     __read_only image2d_t _normal,
										 __read_only image2d_t _specular,
										 float _fresnelPow,
										 float _fresnelCoef,
										 unsigned int _hasDiffuseMap,
										 unsigned int _hasNormalMap,
										 unsigned int _hasSpecularMap,
										 unsigned int _useCornellBox,
										 unsigned int _useExampleSphere,
										 unsigned int _meshInitialised,
										 unsigned int _viewBRDF,
										 unsigned int _hasBRDF,
                     vCamera _cam,
                     unsigned int _w,
                     unsigned int _h,
                     unsigned int _frame,
                     unsigned int _time)
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

    vCamera camera;
    camera.m_origin = _cam.m_origin;
    camera.m_dir = _cam.m_dir;
    camera.m_upV = _cam.m_upV;
    camera.m_rightV = _cam.m_rightV;

    float4 cx = _cam.m_fovScale * _w / _h * camera.m_rightV; // ray direction offset in x direction
    float4 cy = _cam.m_fovScale * camera.m_upV; // ray direction offset in y direction (.5135 is field of view angle)

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
      Ray newcam = createRay(camera.m_origin, normalize(d));

			_colors[ind] += trace(&newcam, _vertices, _normals, _tangents, normalMapvhNodes, _uvs, _hdr, _diffuse, _normal, _specular, _fresnelPow, _fresnelCoef, _hasDiffuseMap, _hasNormalMap, _hasSpecularMap, _useCornellBox, _useExampleSphere, _meshInitialised, _viewBRDF, _hasBRDF, normalMaprdf, &seed0, &seed1) * invSamps;
		}
		float coef = 1.f/_frame;

		write_imagef(_texture, (int2)(x, y), (float4)(pow(clamp(_colors[ind].x * coef, 0.f, 1.f), invGamma),
																									pow(clamp(_colors[ind].y * coef, 0.f, 1.f), invGamma),
																									pow(clamp(_colors[ind].z * coef, 0.f, 1.f), invGamma),
																									1.f));
	}
}
