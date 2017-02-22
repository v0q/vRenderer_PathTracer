#pragma once

#include <cfloat>
#include <vector>

#include "AABB.h"
#include "SBVHNodes.h"
#include "vDataTypes.h"

class SBVH
{
private:
	///
	/// \brief The NodeSpec struct
	///
	struct NodeSpec
	{
		NodeSpec() : m_numRef(0) {}
		///
		/// \brief m_numRef How many triangles is the current node referencing
		///
		unsigned int m_numRef;
		///
		/// \brief m_bounds AABB of the current node
		///
		AABB m_bounds;
	};

	///
	/// \brief The TriRef struct
	///
	struct TriRef
	{
		TriRef() : m_triIndex(-1) {}
		///
		/// \brief m_triIndex Index of the triangle referenced
		///
		int m_triIndex;

		///
		/// \brief m_bounds AABB of the triangle
		///
		AABB m_bounds;
	};

	///
	/// \brief The ObjectSplitCandidate struct
	///
	struct ObjectSplitCandidate
	{
		ObjectSplitCandidate() : m_axis(-1), m_leftTris(0), m_cost(FLT_MAX) {}

		int m_axis;
		int m_leftTris;
		float m_cost;
		AABB m_leftBound;
		AABB m_rightBound;
	};

	///
	/// \brief The SpatialSplitCandidate struct
	///
	struct SpatialSplitCandidate
	{
		SpatialSplitCandidate() : m_axis(-1), m_cost(FLT_MAX), m_location(0.0f) {}

		int m_axis;
		float m_cost;
		float m_location;
	};

	///
	/// \brief The SpatialBin struct
	///
	struct SpatialBin
	{
		AABB m_bounds;
		unsigned int m_enter;
		unsigned int m_exit;
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

	SBVHNode *root() const { return m_root; }
	vHTriangle getTriangle(const unsigned int &_ind) const { return m_triangles[_ind]; }
	ngl::Vec3 getVert(const unsigned int &_ind) const { return m_vertices[_ind]; }
	unsigned int getTriIndex(const unsigned int &_ind) const { return m_triIndices[_ind]; }

private:
	void exec();
	SBVHNode* buildSBVH(const NodeSpec &_nSpec, const unsigned int &_depth);
	SBVHNode* createLeafNode(const NodeSpec &_nSpec);

	// Spatial splitting
	ObjectSplitCandidate findObjectSplitCandidate(const NodeSpec &_nSpec, const float &_nodeCost);
	SpatialSplitCandidate findSpatialSplitCandidate(const NodeSpec &_nSpec, const float &_nodeCost);
	void splitReference(TriRef &o_leftRef, TriRef &o_rightRef, const TriRef &_ref, const unsigned int &_axis, const float &_pos);
	void performSpatialSplit(NodeSpec &io_leftSpec, NodeSpec &io_rightSpec, const NodeSpec &_nSpec, const SpatialSplitCandidate &_split);
	void performObjectSplit(NodeSpec &o_leftSpec, NodeSpec &o_rightSpec, const NodeSpec &_nSpec, const ObjectSplitCandidate &_object);

	// Centroid-based partitioning
	void sortTriRefStack(const unsigned int &_axis, const unsigned int &_first, const unsigned int &_last, bool orig = false);
	int compareBounds(const unsigned int &_axis, const unsigned int &_i, const unsigned int &_j);

	SBVHNode *m_root;

	std::vector<vHTriangle> &m_triangles;
	std::vector<ngl::Vec3> &m_vertices;
	std::vector<TriRef> m_triRefStack;
	std::vector<unsigned int> m_triIndices;

	SpatialBin m_bins[3][kSpatialBins];

	float m_minOverlap;
	bool m_initialised;
};
