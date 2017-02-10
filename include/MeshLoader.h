#pragma once

#include <GL/glew.h>
#include <assimp/scene.h>
#include <string>

typedef struct vFloat3
{
	float x;
	float y;
	float z;
	vFloat3(const float &_x, const float &_y, const float &_z) : x(_x), y(_y), z(_z) {}
} vFloat3;

///
/// \brief The BVH class Simple class for calculating the bounding volume hierarchy for the mesh
///
class BVH
{
public:
	///
	/// @brief vBoundingBox Default ctor
	///
	BVH();

	///
	/// @brief extendBB Simple function to update the extents of the bounding box if needed
	/// @param _vert Mesh vertex to evaluate against the current bounding box
	///
	void computeExtents(const aiVector3t<float> &_vert);
	vFloat3 getSlab(const unsigned int &i) const;
	void print() const;

	static const unsigned int m_numPlaneSetNormals = 7;
	static const vFloat3 m_planeSetNormals[m_numPlaneSetNormals];

private:
	bool m_initialised;
	float m_dNear[m_numPlaneSetNormals];
	float m_dFar[m_numPlaneSetNormals];
	union
	{
		struct
		{
			float m_min;
			float m_max;
		} m_x, m_y, m_z;
	};
};

class vMeshLoader
{
public:
	vMeshLoader(const std::string &_mesh);
	~vMeshLoader();
	static std::vector<vFloat3> loadMesh(const std::string &_mesh);
private:
};
