///
/// \file vRendererCL.h
/// \brief OpenCL implementation of the renderer
/// \authors Teemu Lindborg
/// \version 1.0
/// \date 04/05/17 Updated to NCCA Coding standard
/// Revision History :
/// Initial Version 08/12/16
/// \todo Implement the depth buffer for OpenCL and make sure cleanup is done correctly, especially when overwriting existing buffers
///

#pragma once

#ifdef __APPLE__
	#include <cl/cl.hpp>
#else
	#include <cuda/CL/cl.hpp>
#endif

#include "vRenderer.h"

typedef enum vTextureType {DIFFUSE, NORMAL, SPECULAR} vTextureType;

///
/// \brief The vRendererCL class OpenCL implementation of the renderer
///
class vRendererCL : public vRenderer
{
private:
	///
	/// \brief The vCamera struct OpenCL-style struct for the camera data
	///
	struct vCamera
	{
		///
		/// \brief m_origin Location of the camera
		///
    cl_float4 m_origin;

		///
		/// \brief m_dir Direction the camera is facing
		///
    cl_float4 m_dir;

		///
		/// \brief m_upV Up vector of the camera
		///
    cl_float4 m_upV;

		///
		/// \brief m_rightV Right vector of the camera
		///
    cl_float4 m_rightV;

		///
		/// \brief m_fovScale Field of view scaled used to calculate ray offsets
		///
    float m_fovScale;
  };

public:
	///
	/// \brief vRendererCL Default ctor
	///
  vRendererCL();

	///
	/// \brief ~vRendererCL Default dtor
	///
  ~vRendererCL();

	///
	/// \brief init Creates the OpenCL/OpenGL interop context and initialises buffers that are needed at the beginning. Compiles the trace kernels
	/// \param _w Width of the screen
	/// \param _h Height of the screen
	///
  void init(const unsigned int &_w, const unsigned int &_h) override;

	///
	/// \brief registerTextureBuffer Register the OpenGL texture for OpenCL interop
	/// \param _texture OpenGL texture object
	///
  void registerTextureBuffer(GLuint &_texture) override;

	///
	/// \brief registerDepthBuffer Register the OpenGL depth texture for OpenCL interop, not implemented for now
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
	/// \brief intAsFloat Returns the float equivalent (bit-wise) of a given int
	/// \param _v Int to convert
	/// \return Float of the bit-wise equivalent of the given int
	///
  float intAsFloat(const unsigned int &_v);

	///
	/// \brief m_platform OpenCL platform for the context
	///
  cl::Platform m_platform;

	///
	/// \brief m_device Device used the OpenCL context
	///
  cl::Device m_device;

	///
	/// \brief m_context OpenCL context
	///
  cl::Context m_context;

	///
	/// \brief m_program Tracer program
	///
  cl::Program m_program;

	///
	/// \brief m_kernel Compiled program kernel
	///
  cl::Kernel m_kernel;

	///
	/// \brief m_glTexture Mapped OpenGL texture
	///
  cl::Memory m_glTexture;

	///
	/// \brief m_colorArray OpenCL colour accumulation buffer
	///
  cl::Buffer m_colorArray;

	///
	/// \brief m_queue Command queue used to acquire the GL buffers and enqueue the kernel
	///
  cl::CommandQueue m_queue;

	///
	/// \brief m_GLBuffers Vector containing the GL buffers we want to acquire
	///
	std::vector<cl::Memory> m_GLBuffers;

	///
	/// \brief m_camera Camera for the GPU, device buffer
	///
  vCamera m_camera;

  // Mesh buffers
	///
	/// \brief m_vertices Vertices of a mesh, device buffer
	///
  cl::Buffer m_vertices;

	///
	/// \brief m_normals Normals of a mesh, device buffer
	///
  cl::Buffer m_normals;

	///
	/// \brief m_tangents Tangents of a mesh, device buffer
	///
  cl::Buffer m_tangents;

	///
	/// \brief m_bvhNodes SBVH acceleration structure nodes, device buffer
	///
  cl::Buffer m_bvhNodes;

	///
	/// \brief m_uvs UVs of a mesh, device buffer
	///
  cl::Buffer m_uvs;

	///
	/// \brief m_brdf MERL BRDF data, device buffer
	///
	cl::Buffer m_brdf;

	// Textures
	///
	/// \brief m_hdr HDRI environment map, device buffer
	///
  cl::Image2D m_hdr;

	///
	/// \brief m_diffuse Diffuse texture, device buffer
	///
  cl::Image2D m_diffuse;

	///
	/// \brief m_normal Normal texture, device buffer
	///
  cl::Image2D m_normal;

	///
	/// \brief m_specular Specular texture, device buffer
	///
  cl::Image2D m_specular;

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
	/// \brief m_diffuseMapSet Boolean value to track if a diffuse texture has been loaded
	///
  bool m_diffuseMapSet;

	///
	/// \brief m_normalMapSet Boolean value to track if a normal texture has been loaded
	///
  bool m_normalMapSet;

	///
	/// \brief m_specularMapSet Boolean value to track if a specular texture has been loaded
	///
  bool m_specularMapSet;

	///
	/// \brief m_useCornellBox Whether to use the cornell box scene
	///
	bool m_useCornellBox;

	///
	/// \brief m_useExampleSphere Whether to use the example sphere
	///
	bool m_useExampleSphere;

	///
	/// \brief m_meshInitialised Boolean value to track if a mesh has been loaded
	///
	bool m_meshInitialised;

	///
	/// \brief m_viewBRDF Boolean value to track if the loaded BRDF data should be used
	///
	bool m_viewBRDF;

	///
	/// \brief m_hasBRDF Boolean value to track if BRDF has been loaded to the GPU
	///
	bool m_hasBRDF;

	///
	/// \brief m_initialised Has the renderer been correctly initialised
	///
  bool m_initialised;
};
