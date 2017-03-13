#pragma once

#include <QString>
#include <GL/glew.h>
#include <assimp/scene.h>
#include <string>

#include "SBVH.h"

typedef struct vMeshData
{
	std::vector<vHTriangle> m_triangles;
	std::vector<ngl::Vec3> m_vertices;
	SBVH m_bvh;
  QString m_name;

  vMeshData(const std::vector<vHTriangle> &_tris, const std::vector<ngl::Vec3> &_verts, const SBVH &_bvh, const QString &_name) :
		m_triangles(_tris),
		m_vertices(_verts),
    m_bvh(_bvh),
    m_name(_name)
	{}
} vMeshData;

class vMeshLoader
{
public:
  ~vMeshLoader() {}
	static vMeshData loadMesh(const std::string &_mesh);
private:
  vMeshLoader() {}
  vMeshLoader(const vMeshLoader &_rhs) = delete;
};
