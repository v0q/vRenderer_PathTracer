#pragma once

#include "BVHNodes.h"
#include "vDataTypes.h"

class SBVH
{
private:
	struct TriangleRef
	{
		unsigned int m_triIdx;
		AABB m_bb;
	};

	struct NodeSpec
	{
		unsigned int m_numTris;
		AABB m_bb;
	};

	struct ObjectSplit
	{
		ObjectSplit() :
			m_cost(FLT_MAX), m_splitAxis(-1), m_leftTris(0), m_leftSplit(AABB()), m_rightSplit(AABB()) {}
		float m_cost;
		int m_splitAxis;
		int m_leftTris;
		AABB m_leftSplit;
		AABB m_rightSplit;
	};

	struct SpatialSplit
	{
		SpatialSplit() :
			m_cost(FLT_MAX), m_splitPosition(FLT_MAX), m_axis(-1) {}
		float m_cost;
		float m_splitPosition;
		int m_axis;
	};

	struct SpatialSplitBin
	{
		unsigned int m_entries;
		unsigned int m_exits;
		AABB m_bounds;
	};

public:
	SBVH(vHTriangle *_triangles, ngl::Vec3 *_verts, unsigned int _numTris);
	BVHNode *getRoot() const { return m_root; }
	unsigned int getTriIndex(const unsigned int &_i) const { return m_triIndices[_i]; }

private:
	BVHNode *buildNode(const NodeSpec &_nodeSpec);
	BVHNode *createLeaf(const NodeSpec &_nodeSpec);

	ObjectSplit objectSplit(const NodeSpec &_nodeSpec, const unsigned int &_firstTriRefIndex, const float &_nodeCost);
	SpatialSplit spatialSplit(const NodeSpec &_nodeSpec, const unsigned int &_firstTriRefIndex, const float &_nodeCost);
	void referenceUnsplit(NodeSpec &o_leftSpec, NodeSpec &o_rightSpec, const NodeSpec &_nodeSpec, const SpatialSplit &_spatialSplitCandidate);
	void splitReference(TriangleRef &o_leftSplitRef, TriangleRef &o_rightSplitRef, const TriangleRef &_triRef, const int &_axis, const float &_splitPosition);

	void sortTriRefStack(const unsigned int &_axis, const unsigned int &_first, const unsigned int &_last);
	void insertionSort(const unsigned int &_axis, const unsigned int &_start, const unsigned int &_size);
	int compareBounds(const unsigned int &_axis, const unsigned int &_a, const unsigned int &_b);

	BVHNode *m_root;

	vHTriangle *m_triangles;
	ngl::Vec3 *m_vertices;

	std::vector<unsigned int> m_triIndices;
	std::vector<TriangleRef> m_triangleRefStack;
	std::vector<AABB> m_leftBounds;

	SpatialSplitBin m_bins[kSpatialBins];

	unsigned int m_triangleCount;
	float m_overlapThreshold;
	float m_totalObjectOverlap;
	float m_totalSpatialOverlap;
};
