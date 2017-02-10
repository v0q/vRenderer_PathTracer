#include <iostream>
#include <vector>

#include <cfloat>
#include <GL/glew.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

#include "MeshLoader.h"

const vFloat3 BVH::m_planeSetNormals[BVH::m_numPlaneSetNormals] = {
	vFloat3(1.f, 0.f, 0.f),
	vFloat3(0.f, 1.f, 0.f),
	vFloat3(0.f, 0.f, 1.f),
	vFloat3( sqrtf(3.f) / 3.f,  sqrtf(3.f) / 3.f, sqrtf(3.f) / 3.f),
	vFloat3(-sqrtf(3.f) / 3.f,  sqrtf(3.f) / 3.f, sqrtf(3.f) / 3.f),
	vFloat3(-sqrtf(3.f) / 3.f, -sqrtf(3.f) / 3.f, sqrtf(3.f) / 3.f),
	vFloat3( sqrtf(3.f) / 3.f, -sqrtf(3.f) / 3.f, sqrtf(3.f) / 3.f)
};

BVH::BVH() :
	m_initialised(false)
{
	m_x.m_min = m_x.m_max = 0.f;
	m_y.m_min = m_y.m_max = 0.f;
	m_z.m_min = m_z.m_max = 0.f;

	for(unsigned int i = 0; i < m_numPlaneSetNormals; ++i)
	{
		m_dNear[i] = FLT_MAX;
		m_dFar[i] = -FLT_MAX;
	}
}

void BVH::computeExtents(const aiVector3t<float> &_vert)
{
	for(unsigned int i = 0; i < m_numPlaneSetNormals; ++i)
	{
		float d = m_planeSetNormals[i].x * _vert.x +
							m_planeSetNormals[i].y * _vert.y +
							m_planeSetNormals[i].z * _vert.z;
		m_dNear[i] = std::min(d, m_dNear[i]);
		m_dFar[i] = std::max(d, m_dFar[i]);
	}
//	if(m_initialised)
//	{
//		m_x.m_min = m_x.m_min < _vert.x ? m_x.m_min : _vert.x;
//		m_x.m_max = m_x.m_max > _vert.x ? m_x.m_max : _vert.x;

//		m_y.m_min = m_y.m_min < _vert.y ? m_y.m_min : _vert.y;
//		m_y.m_max = m_y.m_max > _vert.y ? m_y.m_max : _vert.y;

//		m_z.m_min = m_z.m_min < _vert.z ? m_z.m_min : _vert.z;
//		m_z.m_max = m_z.m_max > _vert.z ? m_z.m_max : _vert.z;
//	}
//	else
//	{
//		m_x.m_min = m_x.m_max = _vert.x;
//		m_y.m_min = m_y.m_max = _vert.y;
//		m_z.m_min = m_z.m_max = _vert.z;

//		m_initialised = true;
//	}
}

vFloat3 BVH::getSlab(const unsigned int &i) const
{
	return vFloat3(m_dNear[i], m_dFar[i], 0.0f);
}

void BVH::print() const
{
	for(unsigned int i = 0; i < m_numPlaneSetNormals; ++i)
		std::cout << "Slab " << i+1 << " [" << m_planeSetNormals[i].x << ", " << m_planeSetNormals[i].y << ", " << m_planeSetNormals[i].z << "]\n  Near: " << m_dNear[i] << "\n  Far: " << m_dFar[i] << "\n\n";
}

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
	BVH bb;
	float scale = 15.f;
	float offset = 50.f;

  for(unsigned int i = 0; i < scene->mNumMeshes; ++i)
  {
    aiMesh* mesh = scene->mMeshes[i];
    unsigned int numFaces = mesh->mNumFaces;
    for(unsigned int j = 0; j < numFaces; ++j)
    {
      const aiFace& face = mesh->mFaces[j];
      for(unsigned int k = 0; k < 3; ++k)
      {
				aiVector3t<float> vertex = mesh->mVertices[face.mIndices[k]] * scale;
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
				vertex.x += offset;
				vertex.y += offset / 2;
				vertex.z += offset;

				vertData.push_back(vFloat3(vertex.x, vertex.y, vertex.z));
				vertData.push_back(vFloat3(normal.x, normal.y, normal.z));

				bb.computeExtents(vertex);
      }
		}
  }

	std::cout << "\nBVH Slabs for " << _mesh << ":\n";
	bb.print();
	std::cout << "\n";

	for(unsigned int i = 0; i < BVH::m_numPlaneSetNormals; ++i)
		vertData.push_back(bb.getSlab(i));

  return vertData;
}
