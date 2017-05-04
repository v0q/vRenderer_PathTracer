///
/// \file vRendererCuda.h
/// \brief Cuda implementation of the renderer
/// \authors Teemu Lindborg
/// \version 1.0
/// \date 04/05/17 Updated to NCCA Coding standard
/// Revision History :
/// Initial Version 08/12/16
/// \todo -
///

#pragma once

#include <cuda_runtime.h>
#include "vRenderer.h"
#include "PathTracer.cuh"

///
/// \brief The vRendererCuda class Cuda implementation of the renderer
///
class vRendererCuda : public vRenderer
{
public:
	///
	/// \brief vRendererCuda Default ctor
	///
  vRendererCuda();

	///
	/// \brief ~vRendererCuda Default dtor, calls the cleanup routine
	///
	~vRendererCuda();

	///
	/// \brief init
	/// \param _w
	/// \param _h
	///
  void init(const unsigned int &_w, const unsigned int &_h) override;

	///
	/// \brief registerTextureBuffer Register the OpenGL texture for Cuda interop
	/// \param _texture OpenGL texture object
	///
  void registerTextureBuffer(GLuint &_texture) override;

	///
	/// \brief registerDepthBuffer Register the OpenGL depth texture for Cuda interop
	/// \param _depthTexture OpenGL depth texture object
	///
	void registerDepthBuffer(GLuint &_depthTexture) override;

	///
	/// \brief render Render function, passes the GPU buffers and arguments to the kernel and reserves the needed OpenGL resources and frees them once done
	///
  void render() override;

	///
	/// \brief cleanUp Clears and frees the allocated memory and buffers
	///
  void cleanUp() override;

	///
	/// \brief updateCamera Consumes the updates from the virtual camera and passes the data to the GPU
	///
	void updateCamera() override;

	///
	/// \brief initMesh Takes in 3d mesh data and the SBVH acceleration structure, packs and copies the data to the GPU
	/// \param _sbvhData Structure containing the needed data
	///
	void initMesh(const vMeshData &_meshData) override;

	///
	/// \brief loadHDR Loads an EXR pixelbuffer to the GPU
	/// \param _pixelBuffer Loaded pixelbuffer
	/// \param _w Width of the HDRI map
	/// \param _h Height of the HDRI map
	///
	void loadHDR(const Imf::Rgba *_colours, const unsigned int &_w, const unsigned int &_h) override;

	///
	/// \brief loadTexture Loads a texture to the GPU, performs "inverse gamma correction" if needed as gamma correction is applied at the end of each trace step
	/// \param _texture QImage to read the texture data from
	/// \param _gamma Gamma value of the texture
	/// \param _type Type of the texture, 0 = Diffuse, 1 = Normal, 2 = Specular
	///
	void loadTexture(const QImage &_texture, const float &_gamma, const unsigned int &_type) override;

	///
	/// \brief useBRDF Passes the data to the tracer, whether to use the loaded BRDF data
	/// \param _newVal Whether to use the loaded BRDF data
	///
	void useBRDF(const bool &_newVal) override;

	///
	/// \brief useExampleSphere Passes the info to the tracer, whether to use the example sphere
	/// \param _newVal Whether to use the example sphere
	///
	void useExampleSphere(const bool &_newVal) override;

	///
	/// \brief useCornellBox Passes the info to the tracer, whether to use the cornell box scene or HDRI environment map
	/// \param _newVal Whether to use cornell box or not
	///
	void useCornellBox(const bool &_newVal) override;

	///
	/// \brief clearBuffer Clears the colour buffer, used when something changes in the scene
	///
	void clearBuffer() override;

	///
	/// \brief loadBRDF Loads the BRDF data to the GPU
	/// \param _brdf BRDF data read from a binary file
	/// \return True if the data was loaded correctly
	///
	bool loadBRDF(const float *_brdf) override;

	///
	/// \brief getFrameCount Get the number of frames the current trace has performed
	/// \return The number of frames
	///
  unsigned int getFrameCount() const override { return m_frame - 1; }

private:
	///
	/// \brief validateCuda Validates that a cuda call was successful
	/// \param _err Cuda call that returns its status
	/// \param _msg User message to identify the call in case of an error
	///
	void validateCuda(cudaError_t _err, const std::string &_msg = "");

	///
	/// \brief m_cudaGLTextureBuffer Cuda resource for the GL texture buffer
	///
  cudaGraphicsResource_t m_cudaGLTextureBuffer;

	///
	/// \brief m_cudaGLDepthBuffer Cuda resource for the GL depth texture buffer
	///
	cudaGraphicsResource_t m_cudaGLDepthBuffer;

	///
	/// \brief m_cudaImgArray Cuda array for the GL texture buffer
	///
	cudaArray *m_cudaImgArray;

	///
	/// \brief m_cudaDepthArray Cuda array for the GL depth texture buffer
	///
	cudaArray *m_cudaDepthArray;

	// Cuda buffers
	///
	/// \brief m_camera GPU camera buffer
	///
	vCamera m_camera;

	///
	/// \brief m_colorArray GPU colour accumulation buffer
	///
	float4 *m_colorArray;

	// Mesh buffers
	///
	/// \brief m_vertices GPU mesh vertex buffer
	///
	float4 *m_vertices;

	///
	/// \brief m_normals GPU mesh normal buffer
	///
	float4 *m_normals;

	///
	/// \brief m_tangents GPU mesh tangent buffer
	///
	float4 *m_tangents;

	///
	/// \brief m_bvhData GPU SBVH buffer
	///
	float4 *m_bvhData;

	///
	/// \brief m_uvs GPU mesh uv buffer
	///
	float2 *m_uvs;

	// Textures
	///
	/// \brief m_hdr GPU HDRI texture buffer
	///
	float4 *m_hdr;

	///
	/// \brief m_diffuse GPU diffuse texture buffer
	///
	float4 *m_diffuse;

	///
	/// \brief m_normal GPU normal texture buffer
	///
	float4 *m_normal;

	///
	/// \brief m_specular GPU specular texture buffer
	///
	float4 *m_specular;

	///
	/// \brief m_brdf GPU BRDF data buffer
	///
	float *m_brdf;

	///
	/// \brief m_width Width of the screen/render area
	///
  unsigned int m_width;

	///
	/// \brief m_height Height of the screen/render area
	///
  unsigned int m_height;

	///
	/// \brief m_frame Current frame number
	///
  unsigned int m_frame;

	///
	/// \brief m_initialised Flag to track if the renderer has been correctly initialised and can be used to render
	///
  bool m_initialised;
};
