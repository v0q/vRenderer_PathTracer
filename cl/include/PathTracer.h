///
/// \file PathTracer.h
/// \brief OpenCL device path tracer function definitions and structs
/// \authors Teemu Lindborg
/// \version 1.0
/// \date 04/05/17 Updated to NCCA Coding standard
/// Revision History :
/// Initial Version 08/12/16
/// \todo -
///

#pragma once

typedef enum Refl_t { SPEC, DIFF, BRDF } Refl_t;
typedef enum vTextureType { DIFFUSE, NORMAL, SPECULAR } vTextureType;

///
/// \brief Ray Simple structure for a ray
///
typedef struct Ray
{
	///
	/// \brief m_origin Origin of the current ray
	///
  float4 m_origin;

	///
	/// \brief m_dir Direction of the ray
	///
  float4 m_dir;
} Ray;

///
/// \brief Sphere Simple sphere structure, used for the example sphere, cornell box scene and few others
///
typedef struct Sphere
{
	///
	/// \brief m_r Radius of the sphere
	///
  float m_r;

	///
	/// \brief m_pos Position of the sphere
	///
	float4 m_pos;

	///
	/// \brief m_emission Emission of the sphere
	///
	float4 m_emission;

	///
	/// \brief m_col Colour of the sphere
	///
	float4 m_col;

	///
	/// \brief m_refl Reflective type of the sphere
	///
  Refl_t m_refl;
} Sphere;

///
/// \brief vCamera Simple structure to hold the camera on the GPU
///
typedef struct vCamera
{
	///
	/// \brief m_origin Location of the camera
	///
  float4 m_origin;

	///
	/// \brief m_dir Direction of the camera
	///
  float4 m_dir;

	///
	/// \brief m_upV Up vector of the camera
	///
  float4 m_upV;

	///
	/// \brief m_rightV Right vector of the camera
	///
  float4 m_rightV;

	///
	/// \brief m_fovScale Field of view scale for ray offsets
	///
  float m_fovScale;
} vCamera;

///
/// \brief vHitData Simple structure to contain the details of a ray intersection
///
typedef struct vHitData
{
	///
	/// \brief m_hitPoint Point in space where the intersection happened
	///
	float4 m_hitPoint;

	///
	/// \brief m_normal Normal of the intersected point
	///
	float4 m_normal;

	///
	/// \brief m_tangent Tangent of the intersected point
	///
	float4 m_tangent;

	///
	/// \brief m_emission Emission of the intersected object
	///
	float4 m_emission;

	///
	/// \brief m_color Colour of the intersected point
	///
	float4 m_color;

	///
	/// \brief m_specularColor Specular colour of the intersected point
	///
	float4 m_specularColor;

	///
	/// \brief m_hitType Type of the object hit
	///
	unsigned int m_hitType;
} vHitData;

///
/// \brief floatAsInt Returns the bit-wise equal value as integer
/// \param _a Float to conver
/// \return Bit-wise equal value as an integer
///
unsigned int floatAsInt(const float _a);

///
/// \brief createRay Builds a ray given an origin and a direction
/// \param _o Origin of the ray
/// \param _d Direction of the ray
/// \return Resulting ray
///
Ray createRay(float4 _o, float4 _d);

///
/// \brief get_random Gets a random value between 0.f and 1.f given two seeds, also modifies the seeds
/// \param io_seed0 First seed
/// \param io_seed1 Second seed
/// \return Random value between 0.f and 1.f
///
static float get_random(unsigned int *io_seed0, unsigned int *io_seed1);

///
/// \brief intersectScene Check the intersections of a given ray with the scene, checks for the intersections with the triangle meshes and spheres
/// \param _ray Ray to check intersections against
/// \param _vertices Vertices of a triangle mesh
/// \param _normals Normals of a triangle mesh
/// \param _tangents Tangents of a triangle mesh
/// \param _bvhData SBVH structure
/// \param _uvs UVs of a triangle mesh
/// \param _diffuse Diffuse texture
/// \param _normal Normal texture
/// \param _specular Specular texture
/// \param _hasDiffuseMap Does diffuse texture exist?
/// \param _hasNormalMap Does normal texture exist?
/// \param _hasSpecularMap Does specular texture exist?
/// \param _useCornellBox Should cornell box or HDRI environment be used
/// \param _useExampleSphere Should the example sphere be used
/// \param _meshInitialised Has a mesh been initialised
/// \param _viewBRDF Should BRDFs be used
/// \param _hitData Structure to store the hit results to
/// \return true if an intersection was found
///
bool intersectScene(const Ray *_ray, __global const float4 *_vertices, __global const float4 *_normals, __global const float4 *_tangents, __global const float4 *_bvhNodes, __global const float2 *_uvs, __read_only image2d_t _diffuse, __read_only image2d_t _normal, __read_only image2d_t _specular, bool _hasDiffuseMap, bool _hasNormalMap, bool _hasSpecularMap, bool _useCornellBox, bool _useExampleSphere, bool _meshInitialised, bool _viewBRDF, vHitData *_hitData);

///
/// \brief trace Performs ray bounces, checks for the intersections and does colour accumulation calculations based on the emission and colour data from the intersections
/// \param _camray Initial ray for the trace
/// \param _vertices Vertices of a triangle mesh
/// \param _normals Normals of a triangle mesh
/// \param _tangents Tangents of a triangle mesh
/// \param _bvhNodes SBVH structure
/// \param _uvs UVs of a triangle mesh
/// \param _hdr HDRI map
/// \param _diffuse Diffuse texture
/// \param _normal Normal texture
/// \param _specular Specular texture
/// \param _fresnelPow Fresnel power
/// \param _fresnelCoef Fresnel coefficient
/// \param _hasDiffuseMap Does diffuse texture exist?
/// \param _hasNormalMap Does normal texture exist?
/// \param _hasSpecularMap Does specular texture exist?
/// \param _useCornellBox Should cornell box or HDRI environment be used
/// \param _useExampleSphere Should the example sphere be used
/// \param _meshInitialised Has a mesh been initialised
/// \param _viewBRDF Should BRDFs be used
/// \param _hasBRDF Does BRDF table exist
/// \param _brdf BRDF table
/// \param _seed0 First initial seed for rng
/// \param _seed1 Second initial seed for rng
///
float4 trace(const Ray* _camray, __global const float4 *_vertices, __global const float4 *_normals, __global const float4 *_tangents, __global const float4 *_bvhNodes, __global const float2 *_uvs, __read_only image2d_t _hdr, __read_only image2d_t _diffuse, __read_only image2d_t _normal, __read_only image2d_t _specular, float _fresnelPow, float _fresnelCoef, bool _hasDiffuseMap, bool _hasNormalMap, bool _hasSpecularMap, bool _useCornellBox, bool _useExampleSphere, bool _meshInitialised, bool _viewBRDF, bool _hasBRDF, __global const float *_brdf, unsigned int *_seed0, unsigned int *_seed1);
