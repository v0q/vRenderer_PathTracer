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

class vRenderer
{
public:
  vRenderer() { std::cout << "Parent ctor called\n"; }
  virtual ~vRenderer() { std::cout << "Parent dtor called\n"; }

  virtual void init(const unsigned int &_w = 0, const unsigned int &_h = 0) = 0;
  virtual void registerTextureBuffer(GLuint &_texture) = 0;
  virtual void registerDepthBuffer(GLuint &_depthTexture) = 0;
  virtual void render() = 0;
  virtual void cleanUp() = 0;
	virtual void updateCamera() = 0;
	virtual void initMesh(const vMeshData &_sbvhData) = 0;
	virtual void loadHDR(const Imf::Rgba *_pixelBuffer, const unsigned int &_w, const unsigned int &_h) = 0;
	virtual void loadTexture(const QImage &_texture, const float &_gamma, const unsigned int &_type) = 0;
	virtual void loadBRDF(const float *_brdf) = 0;
	virtual void useBRDF(const bool &_newVal) = 0;
	virtual void useExampleSphere(const bool &_newVal) = 0;
	virtual void useCornellBox(const bool &_newVal) = 0;
  virtual void clearBuffer() = 0;
  virtual unsigned int getFrameCount() const = 0;

	void setFresnelCoef(const float &_newVal) { m_fresnelCoef = _newVal; clearBuffer(); }
	void setFresnelPower(const float &_newVal) { m_fresnelPow = _newVal; clearBuffer(); }
  void setCamera(Camera *_cam) { m_virtualCamera = _cam; updateCamera(); }

protected:
	Camera *m_virtualCamera;
	float m_fresnelCoef;
	float m_fresnelPow;
};
