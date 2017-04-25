#include <iostream>
#include <vector>

#include <GL/glew.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

#include "MeshLoader.h"

vMeshData vMeshLoader::loadMesh(const std::string &_mesh)
{
  Assimp::Importer importer;
//	const aiScene* scene = importer.ReadFile(_mesh.c_str(), aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
	const aiScene* scene = importer.ReadFile(_mesh.c_str(), aiProcessPreset_TargetRealtime_MaxQuality);
	if(!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
  {
    std::cerr << "Failed to load mesh: " << _mesh << "\n";
		std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
		exit(EXIT_FAILURE);
  }

  QString meshName;
	std::vector<ngl::Vec3> vertices;
	std::vector<float> uvs;
	std::vector<vHTriangle> triangles;
	float scale = 1.f;

	for(unsigned int i = 0; i < scene->mNumMeshes; ++i)
	{
		if(scene->mNumMeshes != 1)
			i = scene->mNumMeshes - 1;
		std::cout << i << "\n";
    aiMesh* mesh = scene->mMeshes[i];
    meshName = mesh->mName.C_Str();
		std::cout << "Mesh: " << meshName.toStdString() << "\n";
    unsigned int numFaces = mesh->mNumFaces;
		unsigned int numVerts = mesh->mNumVertices;

		// Mesh center
    ngl::Vec3 center(0.f, 0.f, 0.f);

		triangles.resize(numFaces);
		vertices.resize(numVerts);

		for(unsigned int j = 0; j < numVerts; ++j)
		{
			const aiVector3t<float> vert = mesh->mVertices[j] * scale;
			vertices[j] = ngl::Vec3(vert.x, vert.y, vert.z);
			center += vertices[j];

//			std::cout << mesh->mTextureCoords[j]->x << ", " << mesh->mTextureCoords[j]->y << "\n";

//			uvs.push_back(mesh->mTextureCoords[j]->x);
//			uvs.push_back(mesh->mTextureCoords[j]->y);
		}

		center /= numVerts;

		for(unsigned int j = 0; j < numVerts; ++j)
		{
			vertices[j] -= center;
//			vertices[j].m_z -= 250.f;
		}

		for(unsigned int j = 0; j < numFaces; ++j)
		{
			const aiFace& face = mesh->mFaces[j];

			triangles[j].m_indices[0] = face.mIndices[0];
			triangles[j].m_indices[1] = face.mIndices[1];
			triangles[j].m_indices[2] = face.mIndices[2];

			ngl::Vec3 e1 = vertices[face.mIndices[1]] - vertices[face.mIndices[0]];
			ngl::Vec3 e2 = vertices[face.mIndices[2]] - vertices[face.mIndices[1]];
			ngl::Vec3 e3 = vertices[face.mIndices[0]] - vertices[face.mIndices[2]];

			if(mesh->mNormals != NULL)
			{
				triangles[j].m_normal = ngl::Vec3(mesh->mNormals[face.mIndices[1]].x, mesh->mNormals[face.mIndices[1]].y, mesh->mNormals[face.mIndices[1]].z);
			}
			else
			{
				// plane of triangle, cross product of edge vectors e1 and e2
				triangles[j].m_normal = e1.cross(e2);

				// choose longest alternative normal for maximum precision
				ngl::Vec3 n1 = e2.cross(e3);
				if(n1.length() > triangles[j].m_normal.length())
				{
					triangles[j].m_normal = n1; // higher precision when triangle has sharp angles
				}

				ngl::Vec3 n2 = e3.cross(e1);
				if(n2.length() > triangles[j].m_normal.length())
				{
					triangles[j].m_normal = n2;
				}

				triangles[j].m_normal.normalize();
			}
		}
	}

	SBVH bb(&triangles[0], &vertices[0], triangles.size());
//	exit(0);

  std::cout << meshName.toStdString() << "\n";
  return vMeshData(triangles, vertices, bb, meshName);
}
