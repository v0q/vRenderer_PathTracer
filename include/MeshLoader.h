#pragma once

#include <GL/glew.h>
#include <assimp/scene.h>
#include <string>

#include "SBVH.h"

typedef struct vMeshData
{
	std::vector<vHTriangle> m_triangles;
//	CacheFriendlySBVHNode *m_cfbvh;
	unsigned int *m_cfbvhTriIndices;
	unsigned int m_cfbvhTriIndCount;
	unsigned int m_cfbvhBoxCount;

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
