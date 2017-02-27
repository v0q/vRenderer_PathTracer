#pragma once

#include "BVHNodes.h"
#include "vDataTypes.h"

class BVH
{
public:
	BVH(vHTriangle *_triangles, ngl::Vec3 *_verts, unsigned int _numTris);
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

private:
	BVHNode *buildNode(const NodeSpec &_nodeSpec);
	BVHNode *createLeaf(const NodeSpec &_nodeSpec);
	void sortTriRefStack(const unsigned int &_axis, const unsigned int &_first, const unsigned int &_last);
	void insertionSort(const unsigned int &_axis, const unsigned int &_start, const unsigned int &_size);
	int compareBounds(const unsigned int &_axis, const unsigned int &_a, const unsigned int &_b);

	BVHNode *m_root;

	vHTriangle *m_triangles;
	ngl::Vec3 *m_vertices;

	std::vector<unsigned int> m_triIndices;
	std::vector<TriangleRef> m_triangleRefStack;
	std::vector<AABB> m_leftBounds;

	unsigned int m_triangleCount;
};
