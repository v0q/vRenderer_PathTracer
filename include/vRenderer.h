#pragma once

#include <GL/glew.h>
#include <vector>
#include <assert.h>

#include "MeshLoader.h"

class vRenderer
{
public:
  vRenderer() { std::cout << "Parent ctor called\n"; }
  virtual ~vRenderer() { std::cout << "Parent dtor called\n"; }

  virtual void init(const unsigned int &_w = 0, const unsigned int &_h = 0) = 0;
  virtual void registerTextureBuffer(GLuint &_texture) = 0;
  virtual void render() = 0;
  virtual void cleanUp() = 0;
  virtual void updateCamera(const float *_cam = nullptr, const float *_dir = nullptr) = 0;
  virtual void initMesh(const std::vector<float3> &_vertData) = 0;
  virtual unsigned int getFrameCount() const = 0;
};
