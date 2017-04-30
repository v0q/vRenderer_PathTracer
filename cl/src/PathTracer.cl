#include "cl/include/PathTracer.h"
#include "cl/include/RayIntersection.h"
#include "cl/include/Utilities.h"

__constant float invGamma = 1.f/2.2f;
__constant float invSamps = 1.f/2.f;
__constant unsigned int samps = 2;

__constant Sphere spheres[] = {
  { 3.5f, { 15.f, 0.f, 15.f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f, 0.0f }, SPEC }, // small sphere 1
  { 3.5f, { 25.f, 0.f, 15.f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.4f, 0.4f, 0.4f, 0.0f }, DIFF } // small sphere 2
};

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

bool intersectScene(const Ray *_ray,
                    __global const float4 *_vertices,
                    __global const float4 *_normals,
                    __global const float4 *_tangents,
                    __global const float4 *_bvhNodes,
                    __global const float2 *_uvs,
                    __read_only image2d_t _diffuse,
                    __read_only image2d_t _normal,
                    __read_only image2d_t _specular,
                    bool _hasDiffuseMap,
                    bool _hasNormalMap,
                    bool _hasSpecularMap,
                    vHitData *_hitData)
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
		float dist = intersectSphere(&sphere, _ray);
		/* keep track of the closest intersection and hitobject found so far */
		if(dist != 0.0f && dist < t) {
			t = dist;
			_hitData->m_hitPoint = _ray->m_origin + _ray->m_dir * t;
      _hitData->m_normal = normalize(sphere.m_pos - _hitData->m_hitPoint);
			_hitData->m_color = sphere.m_col;
      _hitData->m_emission = sphere.m_emission;
      _hitData->m_hitType = (int)sphere.m_refl;
      _hitData->m_specularColor = make_float4(0.f, 0.f, 0.f, 0.f);
		}
  }

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
      const float4 n0xy = _bvhNodes[nodeAddr + 0]; // node 0 bounds xy
      const float4 n1xy = _bvhNodes[nodeAddr + 1]; // node 1 bounds xy
      const float4 nz = _bvhNodes[nodeAddr + 2]; // node 0 & 1 bounds z
      float4 tmp = _bvhNodes[nodeAddr + 3]; // Child indices in x & y

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
        if(intersection.x != 0.0f && intersection.x < t)
				{
          t = intersection.x;
					_hitData->m_hitPoint = _ray->m_origin + _ray->m_dir * t;
          _hitData->m_normal = _normals[triAddr];

          float2 uv = (1.f - intersection.y - intersection.z) * _uvs[triAddr] +
                      intersection.y * _uvs[triAddr + 1] +
                      intersection.z * _uvs[triAddr + 2];

          if(_hasDiffuseMap)
          {
            int x = get_image_width(_diffuse) * uv.x;
            int y = get_image_height(_diffuse) * uv.y;
            _hitData->m_color = read_imagef(_diffuse, (int2)(x, y));
          }
          else
          {
            _hitData->m_color = (float4)(1.0f, 1.0f, 1.0f, 0.0f);
          }

          if(_hasNormalMap)
          {
            int x = get_image_width(_normal) * uv.x;
            int y = get_image_height(_normal) * uv.y;
            // Normal map to normals
            float4 normal = normalize((1.f - intersection.y - intersection.z) * _normals[triAddr] +
                                      intersection.y * _normals[triAddr + 1] +
                                      intersection.z * _normals[triAddr + 2]);
            normal.w = 0.f;
            float4 tangent = normalize((1.f - intersection.y - intersection.z) * _tangents[triAddr] +
                                       intersection.y * _tangents[triAddr + 1] +
                                       intersection.z * _tangents[triAddr + 2]);
            tangent.w = 0.f;

            float4 bitangent = cross(normal, tangent);

            float4 normalMap = normalize(2.f * read_imagef(_normal, (int2)(x, y)) - (float4)(1.f, 1.f, 1.f, 0.f));

            // Matrix multiplication TBN (tangent, bitangent, normal) * normal map
            float4 worldSpaceNormal = (float4)(tangent.x * normalMap.x + bitangent.x * normalMap.y + normal.x * normalMap.z,
                                             tangent.y * normalMap.x + bitangent.y * normalMap.y + normal.y * normalMap.z,
                                             tangent.z * normalMap.x + bitangent.z * normalMap.y + normal.z * normalMap.z,
                                             tangent.w * normalMap.x + bitangent.w * normalMap.y + normal.w * normalMap.z + 1.f * normalMap.w);
            _hitData->m_normal = normalize(worldSpaceNormal);
          }
          else
          {
            // Calculate face normal for flat shading
            _hitData->m_normal = normalize(cross(vert0 - vert1, vert0 - vert2));
          }


          if(_hasSpecularMap)
          {
            int x = get_image_width(_specular) * uv.x;
            int y = get_image_height(_specular) * uv.y;
            _hitData->m_specularColor = read_imagef(_specular, (int2)(x, y));
          }
          else
          {
            _hitData->m_specularColor = make_float4(0.f, 0.0f, 0.0f, 0.0f);
          }

					_hitData->m_emission = (float4)(0.f, 0.0f, 0.0f, 0.0f);
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

float4 trace(const Ray* _camray,
             __global const float4 *_vertices,
             __global const float4 *_normals,
             __global const float4 *_tangents,
             __global const float4 *_bvhNodes,
             __global const float2 *_uvs,
             __read_only image2d_t _hdr,
             __read_only image2d_t _diffuse,
             __read_only image2d_t _normal,
             __read_only image2d_t _specular,
             bool _hasDiffuseMap,
             bool _hasNormalMap,
             bool _hasSpecularMap,
             unsigned int *_seed0,
             unsigned int *_seed1)
{
	Ray ray = *_camray;

	float4 accum_color = (float4)(0.0f, 0.0f, 0.0f, 0.f);
	float4 mask = (float4)(1.0f, 1.0f, 1.0f, 0.f);

	for(int bounces = 0; bounces < 4; bounces++)
	{
		vHitData hitData;

		/* if ray misses scene, return background colour */
    if(!intersectScene(&ray, _vertices, _normals, _tangents, _bvhNodes, _uvs, _diffuse, _normal, _specular, _hasDiffuseMap, _hasNormalMap, _hasSpecularMap, &hitData))
		{
			float2 longlat = (float2)(atan2(ray.m_dir.x, ray.m_dir.z), acos(ray.m_dir.y));
			longlat.x = longlat.x < 0 ? longlat.x + 2.0 * PI : longlat.x;
			longlat.x /= 2.0 * PI;
			longlat.y /= PI;

			int2 uv = (int2)(get_image_width(_hdr) * longlat.x, get_image_height(_hdr) * longlat.y);

			accum_color += (mask * 2.0f * read_imagef(_hdr, uv));
      return accum_color;
    }

    // Add the colour and light contributions to the accumulated colour
    accum_color += mask * hitData.m_emission;

    // Next ray's origin is at the hitpoint
    ray.m_origin = hitData.m_hitPoint;
    float4 normal = hitData.m_normal;

    if(hitData.m_hitType == 0)
    {
      ray.m_dir = ray.m_dir - normal * 2.f * dot(normal, ray.m_dir);
      /* add a very small offset to the hitpoint to prevent self intersection */
      ray.m_origin += normal * 0.05f;
    }
    else if(hitData.m_hitType == 1)
    {
      float rouletteRandomFloat = get_random(_seed0, _seed1);
      float threshold = hitData.m_specularColor.x;
      float4 specularColor = hitData.m_specularColor;  // hard-coded
      bool reflectFromSurface = (rouletteRandomFloat < threshold); //computeFresnel(make_Vec3f(n.x, n.y, n.z), incident, incidentIOR, transmittedIOR, reflectionDirection, transmissionDirection).reflectionCoefficient);

      float4 newdir;
      float4 w = normal;
      float4 axis = fabs(w.x) > 0.1f ? (float4)(0.0f, 1.0f, 0.0f, 0.f) : (float4)(1.0f, 0.0f, 0.0f, 0.f);

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
        float rand1 = 2.0f * PI * get_random(_seed0, _seed1);
        float rand2 = get_random(_seed0, _seed1);
        float rand2s = sqrt(rand2);

        /* create a local orthogonal coordinate frame centered at the hitpoint */
        float4 u = normalize(cross(axis, w));
        float4 v = cross(w, u);

        // compute cosine weighted random ray direction on hemisphere
        newdir = normalize(u*cos(rand1)*rand2s + v*sin(rand1)*rand2s + w*sqrt(1 - rand2));

        // multiply mask with colour of object
        mask *= hitData.m_color;
      }

      // offset origin next path segment to prevent self intersection
      ray.m_origin += normal * 0.001f;  // // scene size dependent
      ray.m_dir = newdir;
    }
	}

	return accum_color;
}

__kernel void render(__write_only image2d_t _texture,
                     __global const float4 *_vertices,
                     __global const float4 *_normals,
                     __global const float4 *_tangents,
                     __global const float4 *_bvhNodes,
                     __global const float2 *_uvs,
                     __global float4 *_colors,
                     __read_only image2d_t _hdr,
                     __read_only image2d_t _diffuse,
                     __read_only image2d_t _normal,
                     __read_only image2d_t _specular,
                     unsigned int _hasDiffuseMap,
                     unsigned int _hasNormalMap,
                     unsigned int _hasSpecularMap,
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

      _colors[ind] += trace(&newcam, _vertices, _normals, _tangents, _bvhNodes, _uvs, _hdr, _diffuse, _normal, _specular, _hasDiffuseMap, _hasNormalMap, _hasSpecularMap, &seed0, &seed1) * invSamps;
		}
		float coef = 1.f/_frame;

		write_imagef(_texture, (int2)(x, y), (float4)(pow(clamp(_colors[ind].x * coef, 0.f, 1.f), invGamma),
																									pow(clamp(_colors[ind].y * coef, 0.f, 1.f), invGamma),
																									pow(clamp(_colors[ind].z * coef, 0.f, 1.f), invGamma),
																									1.f));
	}
}
