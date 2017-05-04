///
/// \file MeshLoader.h
/// \brief Simple mesh loader, uses Assimp to load in a 3d model and passes the data to the SBVH builder
///				 Finally returns the data to be passed to the GPU
/// \authors Teemu Lindborg
/// \version 1.0
/// \date 04/05/17 Updated to NCCA Coding standard
/// Revision History :
/// Initial Version 08/12/16
/// \todo Allow the loading of more than a single model
///

#pragma once

#include <QString>
#include <GL/glew.h>
#include <assimp/scene.h>
#include <string>

#include "SBVH.h"

///
/// \brief Simple structure to hold the data needed by the GPU
///
typedef struct vMeshData
{
	///
	/// \brief m_triangles Triangles of the mesh
	///
	std::vector<vHTriangle> m_triangles;

	///
	/// \brief m_vertices Vertices of the mesh
	///
	std::vector<vHVert> m_vertices;

	///
	/// \brief m_bvh Generated acceleration structure
	///
	SBVH m_bvh;

	///
	/// \brief m_name Name of the mesh
	///
  QString m_name;

	///
	/// \brief vMeshData Simple dtor that sets the member variables correctly
	/// \param _tris Triangles
	/// \param _verts Vertices
	/// \param _bvh SBVH acceleration structure
	/// \param _name Name of the mesh
	///
	vMeshData(const std::vector<vHTriangle> &_tris, const std::vector<vHVert> &_verts, const SBVH &_bvh, const QString &_name) :
		m_triangles(_tris),
		m_vertices(_verts),
    m_bvh(_bvh),
    m_name(_name)
	{}
} vMeshData;

///
/// \brief The vMeshLoader class Mesh loader class that reads in a 3d model using assimp and generates the SBVH acceleration structure
///
class vMeshLoader
{
public:
	///
	/// \brief ~vMeshLoader Default dtor
	///
  ~vMeshLoader() {}

	///
	/// \brief loadMesh Loads the mesh and passes the data to the SBVH builder
	/// \param _mesh Location of the 3d model
	/// \return Data structure containing the acceleration structure and the mesh data
	///
	static vMeshData loadMesh(const std::string &_mesh);

private:
	///
	/// \brief vMeshLoader Currently the default ctor of the class is hidden as we can use a static method to load the data
	///
  vMeshLoader() {}

	///
	/// \brief vMeshLoader Currently the default copy ctor of the class is hidden as we can use a static method to load the data
	/// \param _rhs
	///
  vMeshLoader(const vMeshLoader &_rhs) = delete;
};
