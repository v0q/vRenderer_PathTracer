#pragma once

#include <GL/glew.h>
#include <assimp/scene.h>
#include <string>

#include "SBVH.h"

typedef struct vMeshData
{
	std::vector<vHTriangle> m_triangles;
	std::vector<ngl::Vec3> m_vertices;
	SBVH m_sbvh;

	vMeshData(const std::vector<vHTriangle> &_tris, const std::vector<ngl::Vec3> &_verts, const SBVH &_sbvh) :
		m_triangles(_tris),
		m_vertices(_verts),
		m_sbvh(_sbvh)
	{}
} vMeshData;

class vMeshLoader
{
public:
	vMeshLoader(const std::string &_mesh);
	~vMeshLoader();
	static vMeshData loadMesh(const std::string &_mesh);
private:
};
