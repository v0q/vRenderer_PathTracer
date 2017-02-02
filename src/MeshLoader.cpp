#include <iostream>
#include <vector>

#include <GL/glew.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "MeshLoader.h"

vMeshLoader::vMeshLoader(const std::string &_mesh)
{
}

vMeshLoader::~vMeshLoader()
{
}

std::vector<vFloat3> vMeshLoader::loadMesh(const std::string &_mesh)
{
  Assimp::Importer importer;
  const aiScene* scene = importer.ReadFile(_mesh.c_str(), aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
	if(!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
  {
    std::cerr << "Failed to load mesh: " << _mesh << "\n";
		std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
		return std::vector<vFloat3>();
  }

	std::vector<vFloat3> vertData;

  for(unsigned int i = 0; i < scene->mNumMeshes; ++i)
  {
    aiMesh* mesh = scene->mMeshes[i];
    unsigned int numFaces = mesh->mNumFaces;
    for(unsigned int j = 0; j < numFaces; ++j)
    {
      const aiFace& face = mesh->mFaces[j];
      for(unsigned int k = 0; k < 3; ++k)
      {
        aiVector3t<float> vertex = mesh->mVertices[face.mIndices[k]];
        aiVector3t<float> normal;
        if(mesh->mNormals != NULL)
          normal = mesh->mNormals[face.mIndices[k]];
        else
        {
          normal = aiVector3t<float>(mesh->mVertices[face.mIndices[0]].y*mesh->mVertices[face.mIndices[1]].z - mesh->mVertices[face.mIndices[0]].z*mesh->mVertices[face.mIndices[1]].z,
                                     mesh->mVertices[face.mIndices[0]].z*mesh->mVertices[face.mIndices[1]].x - mesh->mVertices[face.mIndices[0]].x*mesh->mVertices[face.mIndices[1]].z,
                                     mesh->mVertices[face.mIndices[0]].x*mesh->mVertices[face.mIndices[1]].y - mesh->mVertices[face.mIndices[0]].y*mesh->mVertices[face.mIndices[1]].x);
          normal.Normalize();
        }
				vertData.push_back(vFloat3(vertex.x, vertex.y, vertex.z));
				vertData.push_back(vFloat3(normal.x, normal.y, normal.z));
      }
		}
  }

  return vertData;
}
