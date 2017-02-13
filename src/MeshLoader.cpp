#include <iostream>
#include <vector>

#include <GL/glew.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

#include "MeshLoader.h"
#include "Utilities.h"

vMeshLoader::vMeshLoader(const std::string &_mesh)
{
}

vMeshLoader::~vMeshLoader()
{
}

vMeshData vMeshLoader::loadMesh(const std::string &_mesh)
{
  Assimp::Importer importer;
  const aiScene* scene = importer.ReadFile(_mesh.c_str(), aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
	if(!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
  {
    std::cerr << "Failed to load mesh: " << _mesh << "\n";
		std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
		return vMeshData();
  }

	std::vector<vHVert> vertices;
	std::vector<vHTriangle> triangles;
	BVH bb;
	float scale = 15.f;
	float offset = 50.f;

  for(unsigned int i = 0; i < scene->mNumMeshes; ++i)
  {
    aiMesh* mesh = scene->mMeshes[i];
    unsigned int numFaces = mesh->mNumFaces;
		unsigned int numVerts = mesh->mNumVertices;

		triangles.resize(numFaces);
		vertices.resize(numVerts);

		for(unsigned int j = 0; j < numVerts; ++j)
		{
			const aiVector3t<float> vert = mesh->mVertices[j] * scale;
			const aiVector3t<float> normal = mesh->mNormals[j];
			vHVert v;
			v.m_vert = vFloat3(vert.x, vert.y, vert.z);
			v.m_normal = vFloat3(normal.x, normal.y, normal.z);
			vertices[j] = v;
		}

		for(unsigned int j = 0; j < numFaces; ++j)
		{
			const aiFace& face = mesh->mFaces[j];

			triangles[j].m_bottom = vFloat3(FLT_MAX, FLT_MAX, FLT_MAX);
			triangles[j].m_top = vFloat3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
			triangles[j].m_indices[0] = face.mIndices[0];
			triangles[j].m_indices[1] = face.mIndices[1];
			triangles[j].m_indices[2] = face.mIndices[2];

			vFloat3 vert1 = vertices[triangles[j].m_indices[1]].m_vert - vertices[triangles[j].m_indices[0]].m_vert;
			vFloat3 vert2 = vertices[triangles[j].m_indices[2]].m_vert - vertices[triangles[j].m_indices[1]].m_vert;
			vFloat3 vert3 = vertices[triangles[j].m_indices[0]].m_vert - vertices[triangles[j].m_indices[2]].m_vert;

			triangles[j].m_center = vFloat3((vert1.x + vert2.x + vert3.x) / 3.0f,
																			(vert1.y + vert2.y + vert3.y) / 3.0f,
																			(vert1.z + vert2.z + vert3.z) / 3.0f);

			if(mesh->mNormals != NULL)
			{
				triangles[j].m_normal = vFloat3(mesh->mNormals[face.mIndices[0]].x, mesh->mNormals[face.mIndices[0]].y, mesh->mNormals[face.mIndices[0]].z);
			}
			else
			{
				// plane of triangle, cross product of edge vectors vert1 and vert2
				triangles[j].m_normal = vUtilities::cross(vert1, vert2);

				// choose longest alternative normal for maximum precision
				vFloat3 n1 = vUtilities::cross(vert2, vert3);
				if(n1.length() > triangles[j].m_normal.length())
					triangles[j].m_normal = n1; // higher precision when triangle has sharp angles

				vFloat3 n2 = vUtilities::cross(vert3, vert1);
				if(n2.length() > triangles[j].m_normal.length())
					triangles[j].m_normal = n2;

				triangles[j].m_normal.normalize();
			}

			// precompute dot product between normal and first triangle vertex
			triangles[j].m_d = vUtilities::dot(triangles[j].m_normal, vertices[triangles[j].m_indices[0]].m_vert);

			// edge planes
			triangles[j].m_e1 = vUtilities::cross(triangles[j].m_normal, vert1);
			triangles[j].m_e1.normalize();

			triangles[j].m_d1 = vUtilities::dot(triangles[j].m_e1, vertices[triangles[j].m_indices[0]].m_vert);

			triangles[j].m_e2 = vUtilities::cross(triangles[j].m_normal, vert2);
			triangles[j].m_e2.normalize();

			triangles[j].m_d2 = vUtilities::dot(triangles[j].m_e2, vertices[triangles[j].m_indices[1]].m_vert);

			triangles[j].m_e3 = vUtilities::cross(triangles[j].m_normal, vert3);
			triangles[j].m_e3.normalize();

			triangles[j].m_d3 = vUtilities::dot(triangles[j].m_e3, vertices[triangles[j].m_indices[2]].m_vert);
		}
	}

	vMeshData meshData;
	meshData.m_triangles = triangles;
	meshData.m_vertices = vertices;
	meshData.m_cfbvh = bb.createBVH(vertices, triangles);

	return meshData;
}
