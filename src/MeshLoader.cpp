///
/// \file MeshLoader.cpp
/// \brief Loads 3D models using Assimp
///

#include <iostream>
#include <vector>

#include <GL/glew.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

#include "MeshLoader.h"

vMeshData vMeshLoader::loadMesh(const std::string &_mesh)
{
	Assimp::Importer importer;

	const aiScene* scene = importer.ReadFile(_mesh.c_str(), aiProcessPreset_TargetRealtime_MaxQuality);
	if(!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
  {
    std::cerr << "Failed to load mesh: " << _mesh << "\n";
		std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
		exit(EXIT_FAILURE);
  }

  QString meshName;
	std::vector<vHVert> vertices;
	std::vector<vHTriangle> triangles;
	float scale = 1.f;

	// Loop through the meshes, need to rework this to allow for loading of more than a single mesh
	for(unsigned int i = 0; i < scene->mNumMeshes; ++i)
	{
		if(scene->mNumMeshes != 1)
			i = scene->mNumMeshes - 1;

    aiMesh* mesh = scene->mMeshes[i];
		meshName = mesh->mName.C_Str();

    unsigned int numFaces = mesh->mNumFaces;
		unsigned int numVerts = mesh->mNumVertices;

		// Mesh center
    ngl::Vec3 center(0.f, 0.f, 0.f);

		triangles.resize(numFaces);
		vertices.resize(numVerts);

		for(unsigned int j = 0; j < numVerts; ++j)
		{
			// Get the data needed by the tracer
			const aiVector3t<float> vert = mesh->mVertices[j] * scale;
			const aiVector3t<float> normal = mesh->mNormals[j];
			vertices[j].m_vert = ngl::Vec3(vert.x, vert.y, vert.z);
			vertices[j].m_normal = ngl::Vec3(normal.x, normal.y, normal.z);

			if(mesh->HasTangentsAndBitangents())
			{
				const aiVector3t<float> tangent = mesh->mTangents[j];
				vertices[j].m_tangent = ngl::Vec3(tangent.x, tangent.y, tangent.z);
			}

			if(mesh->HasTextureCoords(0))
			{
				vertices[j].m_u = mesh->mTextureCoords[0][j].x;
				vertices[j].m_v = 1.f - mesh->mTextureCoords[0][j].y;
			}

			// Calculate the center position of the mesh
			center += vertices[j].m_vert;
		}

		center /= numVerts;

		// Move each vertex so that the center of the mesh is located at the origin
		for(unsigned int j = 0; j < numVerts; ++j)
		{
			vertices[j].m_vert -= center;
		}

		// Get the vertex indices of each triangle
		for(unsigned int j = 0; j < numFaces; ++j)
		{
			const aiFace& face = mesh->mFaces[j];

			triangles[j].m_indices[0] = face.mIndices[0];
			triangles[j].m_indices[1] = face.mIndices[1];
			triangles[j].m_indices[2] = face.mIndices[2];
		}
	}

	// Construct the SBVH
	SBVH bb(&triangles[0], &vertices[0], triangles.size());

  return vMeshData(triangles, vertices, bb, meshName);
}
