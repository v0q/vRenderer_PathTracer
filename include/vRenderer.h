///
/// \file vRenderer.h
/// \brief Abstract base class for the renderer
/// \authors Teemu Lindborg
/// \version 1.0
/// \date 04/05/17 Updated to NCCA Coding standard
/// Revision History :
/// Initial Version 08/12/16
/// \todo -
///

#pragma once

#include <GL/glew.h>
#include <vector>
#include <assert.h>
#include <QImage>
#include <OpenEXR/ImfRgba.h>

#include "Camera.h"
#include "MeshLoader.h"

#define BRDF_SAMPLING_RES_THETA_H       90
#define BRDF_SAMPLING_RES_THETA_D       90
#define BRDF_SAMPLING_RES_PHI_D         360

///
/// \brief The vRenderer class Abstract base class for the path tracer
///
class vRenderer
{
public:
	///
	/// \brief vRenderer Default ctor
	///
  vRenderer() { std::cout << "Parent ctor called\n"; }

	///
	/// \brief ~vRenderer Default virtual dtor
	///
  virtual ~vRenderer() { std::cout << "Parent dtor called\n"; }

	///
	/// \brief init Used to initialise everything that the renderer needs before it can render
	/// \param _w Width of the screen
	/// \param _h Height of the screen
	///
  virtual void init(const unsigned int &_w = 0, const unsigned int &_h = 0) = 0;

	///
	/// \brief registerTextureBuffer Register the OpenGL texture for Cuda/OpenCL interop
	/// \param _texture OpenGL texture object
	///
  virtual void registerTextureBuffer(GLuint &_texture) = 0;

	///
	/// \brief registerDepthBuffer Register the OpenGL depth texture for Cuda/OpenCL interop
	/// \param _depthTexture OpenGL depth texture object
	///
  virtual void registerDepthBuffer(GLuint &_depthTexture) = 0;

	///
	/// \brief render Main render function, maps and reserves the needed opengl buffers once available, performs few tracing samples and frees the mapped buffers for drawing
	///
  virtual void render() = 0;

	///
	/// \brief cleanUp Cleans the allocated memory and GPU buffers
	///
  virtual void cleanUp() = 0;

	///
	/// \brief updateCamera Consumes the updates from the virtual camera and passes the data to the GPU
	///
	virtual void updateCamera() = 0;

	///
	/// \brief initMesh Takes in 3d mesh data and the SBVH acceleration structure, packs and copies the data to the GPU
	/// \param _sbvhData Structure containing the needed data
	///
	virtual void initMesh(const vMeshData &_sbvhData) = 0;

	///
	/// \brief loadHDR Loads an EXR pixelbuffer to the GPU
	/// \param _pixelBuffer Loaded pixelbuffer
	/// \param _w Width of the HDRI map
	/// \param _h Height of the HDRI map
	///
	virtual void loadHDR(const Imf::Rgba *_pixelBuffer, const unsigned int &_w, const unsigned int &_h) = 0;

	///
	/// \brief loadTexture Loads a texture to the GPU, performs "inverse gamma correction" if needed as gamma correction is applied at the end of each trace step
	/// \param _texture QImage to read the texture data from
	/// \param _gamma Gamma value of the texture
	/// \param _type Type of the texture, 0 = Diffuse, 1 = Normal, 2 = Specular
	///
	virtual void loadTexture(const QImage &_texture, const float &_gamma, const unsigned int &_type) = 0;

	///
	/// \brief useBRDF Passes the data to the tracer, whether to use the loaded BRDF data
	/// \param _newVal Whether to use the loaded BRDF data
	///
	virtual void useBRDF(const bool &_newVal) = 0;

	///
	/// \brief useExampleSphere Passes the info to the tracer, whether to use the example sphere
	/// \param _newVal Whether to use the example sphere
	///
	virtual void useExampleSphere(const bool &_newVal) = 0;

	///
	/// \brief useCornellBox Passes the info to the tracer, whether to use the cornell box scene or HDRI environment map
	/// \param _newVal Whether to use cornell box or not
	///
	virtual void useCornellBox(const bool &_newVal) = 0;

	///
	/// \brief clearBuffer Clears the colour buffer, used when something changes in the scene
	///
  virtual void clearBuffer() = 0;

	///
	/// \brief loadBRDF Loads the BRDF data to the GPU
	/// \param _brdf BRDF data read from a binary file
	/// \return True if the data was loaded correctly
	///
	virtual bool loadBRDF(const float *_brdf) = 0;

	///
	/// \brief getFrameCount Get the number of frames the current trace has performed
	/// \return The number of frames
	///
  virtual unsigned int getFrameCount() const = 0;

	///
	/// \brief setFresnelCoef Set the fresnel coefficient for fresnel-like reflections, also clears the colour buffer
	/// \param _newVal New coefficient
	///
	void setFresnelCoef(const float &_newVal) { m_fresnelCoef = _newVal; clearBuffer(); }

	///
	/// \brief setFresnelPower Set the fresnel power for fresnel-life reflections, also clears the colour buffer
	/// \param _newVal New power
	///
	void setFresnelPower(const float &_newVal) { m_fresnelPow = _newVal; clearBuffer(); }

	///
	/// \brief setCamera Initialises the virtual camera for the renderer
	/// \param _cam Virtual camera that the renderer will use
	///
  void setCamera(Camera *_cam) { m_virtualCamera = _cam; updateCamera(); }

protected:
	///
	/// \brief m_virtualCamera Pointer to the camera
	///
	Camera *m_virtualCamera;

	///
	/// \brief m_fresnelCoef Fresnel coefficient
	///
	float m_fresnelCoef;

	///
	/// \brief m_fresnelPow Fresnel power
	///
	float m_fresnelPow;
};
