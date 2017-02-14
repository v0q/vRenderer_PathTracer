#pragma once

///
/// @brief Generates a BVH tree to be used to optimise/speed up GPU path tracing
/// Modified from :-
/// Sam Lapere (January 18, 2016). GPU path tracing tutorial 3: GPU-friendly Acceleration Structures. Now you're cooking with GAS! [online].
/// [Accessed 2017]. Available from: http://raytracey.blogspot.co.uk/2016/01/gpu-path-tracing-tutorial-3-take-your.html & https://github.com/straaljager/GPU-path-tracing-tutorial-3/
/// @author Teemu Lindborg
/// @version 0.1
/// @date 13/01/17 Initial version
/// Revision History :
/// -
/// @todo Tidying up and rewriting
///

#include <cfloat>
#include <vector>

#include "MeshData.h"

struct BoundingBox {
	vFloat3 m_bottom;
	vFloat3 m_top;
	vFloat3 m_center;

	const vHTriangle *m_triangles;

	BoundingBox() :
		m_bottom(FLT_MAX, FLT_MAX, FLT_MAX),
		m_top(-FLT_MAX, -FLT_MAX, -FLT_MAX),
		m_triangles(nullptr)
	{}
};

struct BVHNode {
	vFloat3 m_bottom;
	vFloat3 m_top;
	virtual bool isLeaf() = 0;
};

struct BVHInner : BVHNode {
	BVHNode *m_leftNode;
	BVHNode *m_rightNode;
	virtual bool isLeaf() override { return false; }
};

struct BVHLeaf : BVHNode {
	std::vector<const vHTriangle*> m_triangles;
	virtual bool isLeaf() override { return true; }
};

struct CacheFriendlyBVHNode {
	// bounding box
	vFloat3 m_bottom;
	vFloat3 m_top;

	// parameters for leafnodes and innernodes occupy same space (union) to save memory
	// top bit discriminates between leafnode and innernode
	// no pointers, but indices (int): faster

	union {
		// inner node - stores indexes to array of CacheFriendlyBVHNode
		struct {
			unsigned int m_leftIndex;
			unsigned int m_rightIndex;
		} m_inner;
		// leaf node: stores triangle count and starting index in triangle list
		struct {
			unsigned int m_count; // Top-most bit set, leafnode if set, innernode otherwise
			unsigned int m_startIndexInTriIndexList;
		} m_leaf;
	} m_u;
};

class BVH
{
public:
	BVH() :
		m_triIndices(nullptr),
		m_triCount(0),
		m_boxCount(0),
		m_bottom(FLT_MAX, FLT_MAX, FLT_MAX),
		m_top(-FLT_MAX, -FLT_MAX, -FLT_MAX),
		m_cfBVH(nullptr)
	{}
	~BVH() {}

	CacheFriendlyBVHNode* createBVH(const std::vector<vHVert> &_vertices, const std::vector<vHTriangle> &_triangles);
	unsigned int *getTriIndices() const { return m_triIndices; }
	unsigned int getBoxCount() const { return m_boxCount; }
	unsigned int getTriIndCount() const { return m_triCount; }
private:
	void createCFBVH(BVHNode* root, const std::vector<vHVert> &_vertices, const std::vector<vHTriangle> &_triangles);
	void initCFBVH(BVHNode *root, const vHTriangle *_firstTri, unsigned &_cfBoxCount, unsigned &_cfTriCount);
	BVHNode* recurseBoundingBoxes(const std::vector<BoundingBox> &_workingTree, unsigned int _depth = 0);

	unsigned int *m_triIndices;
	unsigned int m_triCount;
	unsigned int m_boxCount;
	vFloat3 m_bottom;
	vFloat3 m_top;

	CacheFriendlyBVHNode* m_cfBVH;
};
