#include <iostream>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "MeshLoader.h"

vMeshLoader::vMeshLoader(const std::string &_mesh)
{
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(_mesh.c_str(), aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
	if(!scene)
	{
		std::cerr << "Failed to load mesh: " << _mesh << "\n";
		return;
	}

	const int vSize = sizeof(aiVector3D)*2 + sizeof(aiVector2D);

	for(unsigned int i = 0; i < scene->mNumMeshes; ++i)
	{
		aiMesh* mesh = scene->mMeshes[i];
		int numFaces = mesh->mNumFaces;
//		int size = vboModelData.GetCurrentSize();
//		m_startIndices.push_back(size/vSize);
		for(unsigned int j = 0; j < numFaces; ++j)
		{
			const aiFace& face = mesh->mFaces[j];
			for(unsigned int k = 0; k < 3; ++k)
			{
				aiVector3D pos = mesh->mVertices[face.mIndices[k]];
				aiVector3D uv = mesh->mTextureCoords[0][face.mIndices[k]];
				aiVector3D normal = mesh->HasNormals() ? mesh->mNormals[face.mIndices[k]] : aiVector3D(1.0f, 1.0f, 1.0f);

				// Add data to VBO;
			}
		}
	}
}

vMeshLoader::~vMeshLoader()
{

}
