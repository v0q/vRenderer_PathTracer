///
/// \file PathTracer.cuh
/// \brief CPU-GPU header file for the path tracer
/// \authors Teemu Lindborg
/// \version 1.0
/// \date 04/05/17 Updated to NCCA Coding standard
/// Revision History :
/// Initial Version 08/12/16
/// \todo -
///

#pragma once

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

typedef enum vTextureType { DIFFUSE, NORMAL, SPECULAR } vTextureType;

///
/// \brief cu_runRenderKernel CPU function that calls the GPU kernel
/// \param o_texture GL texture to write the result to
/// \param o_depth GL depth texture to write the depth to
/// \param _hdr HDRI map
/// \param _vertices Mesh vertices
/// \param _normals Mesh normals
/// \param _tangents Mesh tangents
/// \param _bvhData SBVH acceleration structure
/// \param _uvs Mesh UVs
/// \param io_colorArr Colour accumulation buffer
/// \param _cam Camera
/// \param _w Width of the screen/render area
/// \param _h Height of the screen/render area
/// \param _frame Current frame number
/// \param _time Elapsed time (used as a seed for rng)
/// \param _fresnelCoef Fresnel coefficient
/// \param _fresnelPow Fresnel power
///
void cu_runRenderKernel(cudaSurfaceObject_t o_texture, cudaSurfaceObject_t o_depth, float4 *_hdr, float4 *_vertices, float4 *_normals, float4 *_tangents, float4 *_bvhData, float2 *_uvs, float4 *io_colorArr, vCamera _cam, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time, float _fresnelCoef, float _fresnelPow);

///
/// \brief cu_bindTexture Binds a given buffer to a cuda texture
/// \param _deviceTexture Device texture buffer
/// \param _w Width of the texture
/// \param _h Height of the texture
/// \param _type Type of the texture, 0 = Diffuse, 1 = Normal, 2 = Specular
///
void cu_bindTexture(const float4 *_deviceTexture, const unsigned int _w, const unsigned int _h, const vTextureType &_type);

///
/// \brief cu_bindBRDF Binds the BRDF data to a cuda texture
/// \param _brdf Device brdf buffer
///
void cu_bindBRDF(const float *_brdf);

///
/// \brief cu_useBRDF Toggles a device symbol for BRDF usage
/// \param _newVal New value, whether to use BRDF or not
///
void cu_useBRDF(const bool &_newVal);

///
/// \brief cu_useExampleSphere Toggles a device symbol for example sphere usage
/// \param _newVal New value, whether to use the example sphere or not
///
void cu_useExampleSphere(const bool &_newVal);

///
/// \brief cu_useCornellBox Toggles a device symbol for whether to use the cornell box or HDRI
/// \param _newVal New value, whether to use cornell box or not
///
void cu_useCornellBox(const bool &_newVal);

///
/// \brief cu_setHDRDim Set the dimensions of the HDRI map, as it's stored in a float4 buffer rather than a texture
/// \param _w Width of the HDRI map
/// \param _h Height of the HDRI map
///
void cu_setHDRDim(const unsigned int &_w, const unsigned int &_h);

///
/// \brief cu_meshInitialised Propagate the info that a mesh has been initialised to the GPU
///
void cu_meshInitialised();

///
/// \brief cu_fillFloat4 Fill a device float4 pointer with a given value
/// \param _dPtr Device pointer
/// \param _val Value to fill the pointer with
/// \param _size Size of the device pointer
///
void cu_fillFloat4(float4 *_dPtr, const float4 _val, const unsigned int _size);

///
/// \brief cu_cleanUp Cleans up the textures etc from the device
///
void cu_cleanUp();
