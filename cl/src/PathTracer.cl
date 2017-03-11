#include "cl/include/PathTracer.h"
#include "cl/include/RayIntersection.h"
#include "cl/include/Utilities.h"

#define BVH_MAX_STACK_SIZE 32
__constant float invGamma = 1.f/2.2f;
__constant unsigned int samps = 2;
__constant float invSamps = 1.f/2;

__constant Sphere spheres[] = {
//	{ 1e5f, { 50.0f, 1e5f - 40.f, 81.6f, 0.0f },							{ 0.0f, 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f, 0.0f } }, //Botm
//	{ 16.5f, { 27.0f, 16.5f, 47.0f, 0.0f },						{ 0.0f, 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f, 0.0f } }, // small sphere 1
//	{ 16.5f, { 73.0f, 16.5f, 78.0f, 0.0f },						{ 0.0f, 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f, 0.0f } }, // small sphere 2
  { 1e5f,   { 0.f, 0.f, 0.f, 0.0f },    { 0.8f, 0.8, 0.8, 0.0f }, { 0.f, 0.f, 0.f, 0.f } }, //Botm
//  { 150.0f, { 50.0f, 300.6f - .77f, 81.6f, 0.0f },  { 2.8f, 1.8f, 1.6f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } }  // Light
};

Ray createRay(float4 _o, float4 _d)
{
	Ray ray;
	ray.m_origin = _o;
  ray.m_dir = _d;
	return ray;
}

bool intersectScene(const Ray *_ray, __global const float4 *_vertices, __global const float4 *_normals, __global const float4 *_bvhNodes, vHitData *_hitData)
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

//      float longlatX = atan2(_ray->m_dir.x, _ray->m_dir.z); // Y is up, swap x for y and z for x
//      longlatX = longlatX < 0.f ? longlatX + 2*PI : longlatX;  // wrap around full circle if negative
//      float longlatY = acos(_ray->m_dir.y); // add RotateMap at some point, see Fragmentarium

//      // map theta and phi to u and v texturecoordinates in [0,1] x [0,1] range
//      float offsetY = 0.5f;
//      float u = longlatX / 2*PI; // +offsetY;
//      float v = longlatY / PI;
      float u = atan2(_hitData->m_normal.x, _hitData->m_normal.z) / (2*PI) + 0.5;
      float v = _hitData->m_normal.y * 0.5 + 0.5;

      _hitData->m_uv = (float2)(u, v);
      _hitData->m_type = 0;
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

			int2 indices = (int2)(floatAsInt(tmp.x), floatAsInt(tmp.y));

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

				float dist = intersectTriangle(vert0, vert1, vert2, _ray);
				if(dist != 0.0f && dist < t)
				{
					t = dist;
					_hitData->m_hitPoint = _ray->m_origin + _ray->m_dir * t;
          _hitData->m_normal = _normals[triAddr];
          _hitData->m_color = (float4)(1.0f, 1.0f, 1.0f, 0.0f);
					_hitData->m_emission = (float4)(0.f, 0.0f, 0.0f, 0.0f);
          _hitData->m_type = 1;
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

//float4 trace(const Ray* _camray, __read_only image1d_t _vertices, __read_only image1d_t _normals, __read_only image1d_t _bvhNodes, __read_only image1d_t _triIdxList, unsigned int *_seed0, unsigned int *_seed1)
float4 trace(const Ray* _camray, __global const float4 *_vertices, __global const float4 *_normals, __global const float4 *_bvhNodes, __read_only image2d_t _hdr, unsigned int *_seed0, unsigned int *_seed1)
{
	Ray ray = *_camray;

	float4 accum_color = (float4)(0.0f, 0.0f, 0.0f, 0.f);
	float4 mask = (float4)(1.0f, 1.0f, 1.0f, 0.f);

	for(int bounces = 0; bounces < 4; bounces++)
	{
		vHitData hitData;

		/* if ray misses scene, return background colour */
    if(!intersectScene(&ray, _vertices, _normals, _bvhNodes, &hitData)) {
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
    if(hitData.m_type == 0)
    {
      int2 uv = (int2)(get_image_width(_hdr)*hitData.m_uv.x, get_image_height(_hdr)*hitData.m_uv.y);
      accum_color += (mask * 2.0f * read_imagef(_hdr, uv));
      return accum_color;
    }
    else
    {
      accum_color += mask * hitData.m_emission;
      mask *= hitData.m_color;
    }

    /* the mask colour picks up surface colours at each bounce */

    /* perform cosine-weighted importance sampling for diffuse surfaces*/
    mask *= dot(newdir, normal_facing);
    mask *= 2;
	}

	return accum_color;
}


__kernel void render(__write_only image2d_t _texture, __global const float4 *_vertices, __global const float4 *_normals, __global const float4 *_bvhNodes, __global const unsigned int *_triIdxList,
                     __global float4 *_colors, __read_only image2d_t _hdr, float4 _cam, float4 _dir, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time)
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

      _colors[ind] += trace(&newcam, _vertices, _normals, _bvhNodes, _hdr, &seed0, &seed1) * invSamps;
		}
		float coef = 1.f/_frame;

		write_imagef(_texture, (int2)(x, y), (float4)(pow(clamp(_colors[ind].x * coef, 0.f, 1.f), invGamma),
																									pow(clamp(_colors[ind].y * coef, 0.f, 1.f), invGamma),
																									pow(clamp(_colors[ind].z * coef, 0.f, 1.f), invGamma),
																									1.f));
	}
}
