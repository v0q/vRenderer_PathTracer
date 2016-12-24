#pragma once

#include <GL/glew.h>
#include <string>

typedef struct float3
{
  float x;
  float y;
  float z;
  float3(const float &_x, const float &_y, const float &_z) : x(_x), y(_y), z(_z) {}
} float3;

class vMeshLoader
{
public:
	vMeshLoader(const std::string &_mesh);
	~vMeshLoader();
  static std::vector<float3> loadMesh(const std::string &_mesh);
private:
};
