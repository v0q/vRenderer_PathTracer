#include <iostream>
#include <vector>

#include <GL/glew.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

#include "MeshLoader.h"

vMeshLoader::vMeshLoader(const std::string &_mesh)
{
}

vMeshLoader::~vMeshLoader()
{
}

vMeshData vMeshLoader::loadMesh(const std::string &_mesh)
{
  Assimp::Importer importer;
//	const aiScene* scene = importer.ReadFile(_mesh.c_str(), aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
	const aiScene* scene = importer.ReadFile(_mesh.c_str(), aiProcessPreset_TargetRealtime_MaxQuality);
	if(!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
  {
    std::cerr << "Failed to load mesh: " << _mesh << "\n";
		std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
		return vMeshData();
  }

	std::vector<ngl::Vec3> vertices;
	std::vector<vHTriangle> triangles;
	float scale = 15.f;
	float offset = 50.f;

	std::cout << scene->mNumMeshes << "\n";

  for(unsigned int i = 0; i < scene->mNumMeshes; ++i)
  {
    aiMesh* mesh = scene->mMeshes[i];
    unsigned int numFaces = mesh->mNumFaces;
		unsigned int numVerts = mesh->mNumVertices;

		std::cout << "Num verts: " << numVerts << "\n";

		triangles.resize(numFaces);
		vertices.resize(numVerts);

		for(unsigned int j = 0; j < numVerts; ++j)
		{
			const aiVector3t<float> vert = mesh->mVertices[j] * scale;
			vertices[j] = ngl::Vec3(vert.x + offset, vert.y + offset/2., vert.z + offset);
		}

		for(unsigned int j = 0; j < numFaces; ++j)
		{
			const aiFace& face = mesh->mFaces[j];

			triangles[j].m_indices[0] = face.mIndices[0];
			triangles[j].m_indices[1] = face.mIndices[1];
			triangles[j].m_indices[2] = face.mIndices[2];

			ngl::Vec3 e1 = vertices[triangles[j].m_indices[1]] - vertices[triangles[j].m_indices[0]];
			ngl::Vec3 e2 = vertices[triangles[j].m_indices[2]] - vertices[triangles[j].m_indices[1]];
			ngl::Vec3 e3 = vertices[triangles[j].m_indices[0]] - vertices[triangles[j].m_indices[2]];

			if(mesh->mNormals != NULL)
			{
				triangles[j].m_normal = ngl::Vec3(mesh->mNormals[face.mIndices[0]].x, mesh->mNormals[face.mIndices[0]].y, mesh->mNormals[face.mIndices[0]].z);
			}
			else
			{
				// plane of triangle, cross product of edge vectors e1 and e2
				triangles[j].m_normal = e1.cross(e2);

				// choose longest alternative normal for maximum precision
				ngl::Vec3 n1 = e2.cross(e3);
				if(n1.length() > triangles[j].m_normal.length())
					triangles[j].m_normal = n1; // higher precision when triangle has sharp angles

				ngl::Vec3 n2 = e3.cross(e1);
				if(n2.length() > triangles[j].m_normal.length())
					triangles[j].m_normal = n2;

				triangles[j].m_normal.normalize();
			}
		}
	}

	SBVH bb(triangles, vertices);
	exit(0);
//	bb.createSBVH(vertices, triangles);
//	vMeshData meshData;
//	meshData.m_triangles = triangles;
//	meshData.m_cfbvh = bb.createBVH(vertices, triangles);
//	meshData.m_cfbvhTriIndices = bb.getTriIndices();
//	meshData.m_cfbvhTriIndCount = bb.getTriIndCount();
//	meshData.m_cfbvhBoxCount = bb.getBoxCount();

//	return;
//	return meshData;
}
