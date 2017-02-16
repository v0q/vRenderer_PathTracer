#pragma once

#include <cfloat>
#include <vector>

#include "AABB.h"
#include "SBVHNodes.h"
#include "vDataTypes.h"

class SBVH
{
private:
	struct NodeSpec
	{
		///
		/// \brief m_numRef How many triangles is the current node referencing
		///
		unsigned int m_numRef;
		///
		/// \brief m_bounds AABB of the current node
		///
		AABB m_bounds;
	};

	struct TriRef
	{
		///
		/// \brief m_triIndex Index of the triangle referenced
		///
		unsigned int m_triIndex;

		///
		/// \brief m_bounds AABB of the triangle
		///
		AABB m_bounds;
	};

public:
	SBVH() = delete;
	SBVH(std::vector<vHTriangle> &_tris, std::vector<ngl::Vec3> &_verts) :
		m_triangles(_tris),
		m_vertices(_verts)
	{
		m_initialised = (m_triangles.size() && m_vertices.size());
		exec();
	}
	~SBVH() {}

private:
	void exec();
	SBVHNode* buildSBVH(const NodeSpec &_nSpec, const unsigned int &_depth);
	SBVHNode* createLeafNode(const NodeSpec &_nSpec);

	// Spatial splitting
	void findObjectSplitCandidate(const NodeSpec &_nSpec, const float &_nodeCost);

	// Centroid-based partitioning
	void sortTriRefStack(const unsigned int &_axis, const unsigned int &_first, const unsigned int &_last, bool orig = false);
	int compareBounds(const unsigned int &_axis, const unsigned int &_i, const unsigned int &_j);

	SBVHNode *m_root;

	std::vector<vHTriangle> &m_triangles;
	std::vector<ngl::Vec3> &m_vertices;
	std::vector<TriRef> m_triRefStack;
	std::vector<unsigned int> m_triIndices;

	bool m_initialised;
};
