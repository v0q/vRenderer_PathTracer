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
//	const aiScene* scene = importer.ReadFile(_mesh.c_str(), aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
	const aiScene* scene = importer.ReadFile(_mesh.c_str(), aiProcessPreset_TargetRealtime_MaxQuality);
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
			const aiVector3t<float> normal = mesh->mNormals[j];

			vHVert v;
			v.m_vert = vFloat3(vert.x + offset, vert.y + offset/2., vert.z + offset);
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

			vFloat3 e1 = vertices[triangles[j].m_indices[1]].m_vert - vertices[triangles[j].m_indices[0]].m_vert;
			vFloat3 e2 = vertices[triangles[j].m_indices[2]].m_vert - vertices[triangles[j].m_indices[1]].m_vert;
			vFloat3 e3 = vertices[triangles[j].m_indices[0]].m_vert - vertices[triangles[j].m_indices[2]].m_vert;

			triangles[j].m_center = vFloat3((e1.x + e2.x + e3.x) / 3.0f,
																			(e1.y + e2.y + e3.y) / 3.0f,
																			(e1.z + e2.z + e3.z) / 3.0f);

			for(unsigned int i = 0; i < 3; ++i)
			{
				triangles[j].m_bottom = vUtilities::minvFloat3(triangles[j].m_bottom, vertices[triangles[j].m_indices[i]].m_vert);
				triangles[j].m_top = vUtilities::maxvFloat3(triangles[j].m_top, vertices[triangles[j].m_indices[i]].m_vert);
			}

			if(mesh->mNormals != NULL)
			{
				triangles[j].m_normal = vFloat3(mesh->mNormals[face.mIndices[0]].x, mesh->mNormals[face.mIndices[0]].y, mesh->mNormals[face.mIndices[0]].z);
			}
			else
			{
				// plane of triangle, cross product of edge vectors e1 and e2
				triangles[j].m_normal = vUtilities::cross(e1, e2);

				// choose longest alternative normal for maximum precision
				vFloat3 n1 = vUtilities::cross(e2, e3);
				if(n1.length() > triangles[j].m_normal.length())
					triangles[j].m_normal = n1; // higher precision when triangle has sharp angles

				vFloat3 n2 = vUtilities::cross(e3, e1);
				if(n2.length() > triangles[j].m_normal.length())
					triangles[j].m_normal = n2;

				triangles[j].m_normal.normalize();
			}

			// precompute dot product between normal and first triangle vertex
			triangles[j].m_d = vUtilities::dot(triangles[j].m_normal, vertices[triangles[j].m_indices[0]].m_vert);

			// edge plane
			triangles[j].m_e1 = vUtilities::cross(triangles[j].m_normal, e1);
			triangles[j].m_e1.normalize();

			triangles[j].m_d1 = vUtilities::dot(triangles[j].m_e1, vertices[triangles[j].m_indices[0]].m_vert);

			triangles[j].m_e2 = vUtilities::cross(triangles[j].m_normal, e2);
			triangles[j].m_e2.normalize();

			triangles[j].m_d2 = vUtilities::dot(triangles[j].m_e2, vertices[triangles[j].m_indices[1]].m_vert);

			triangles[j].m_e3 = vUtilities::cross(triangles[j].m_normal, e3);
			triangles[j].m_e3.normalize();

			triangles[j].m_d3 = vUtilities::dot(triangles[j].m_e3, vertices[triangles[j].m_indices[2]].m_vert);
		}
	}

	vMeshData meshData;
	meshData.m_triangles = triangles;
	meshData.m_cfbvh = bb.createBVH(vertices, triangles);
	meshData.m_cfbvhTriIndices = bb.getTriIndices();
	meshData.m_cfbvhTriIndCount = bb.getTriIndCount();
	meshData.m_cfbvhBoxCount = bb.getBoxCount();

	return meshData;
}
