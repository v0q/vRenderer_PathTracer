#pragma once

#include <GL/glew.h>
#include <vector>
#include <assert.h>
#include <OpenEXR/ImfRgba.h>

#include "Camera.h"
#include "MeshLoader.h"

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
	virtual void loadTexture(const unsigned char *_texture, const unsigned int &_w, const unsigned int &_h, const unsigned int &_type) = 0;
  virtual void clearBuffer() = 0;
  virtual unsigned int getFrameCount() const = 0;

  void setCamera(Camera *_cam) { m_virtualCamera = _cam; updateCamera(); }

protected:
	Camera *m_virtualCamera;
};
