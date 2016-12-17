#pragma once

#include <iostream>
#include <GL/glew.h>

class vRenderer
{
public:
  vRenderer() { std::cout << "Parent ctor called\n"; }
  virtual ~vRenderer() { std::cout << "Parent dtor called\n"; }

  virtual void init() = 0;
  virtual void registerTextureBuffer(GLint &_texture) = 0;
  virtual void cleanUp() = 0;
};
