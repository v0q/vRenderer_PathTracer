#pragma once

#include <GL/glew.h>
#include <assimp/scene.h>
#include <string>

#include "BVH.h"

typedef struct vMeshData
{
	std::vector<vHTriangle> m_triangles;
	std::vector<vHVert> m_vertices;
	CacheFriendlyBVHNode *m_cfbvh;

	vMeshData() {}
} vMeshData;

class vMeshLoader
{
public:
	vMeshLoader(const std::string &_mesh);
	~vMeshLoader();
	static vMeshData loadMesh(const std::string &_mesh);
private:
};
