///
/// \file SBVH.h
/// \brief SBVH class encapsulating the building of a Spatial Splits in Bounding Volume Hierarchies acceleration structure.
///				 SBVH aims to minimise the node intersections of a regular BVH. The implementation is based on "Spatial Splits in Bounding Volume Hierarchies"
///				 by Martin Stitch et. al. Available at http://www.nvidia.ca/docs/IO/77714/sbvh.pdf
///
///				 Some implementation decisions are adapted from the GPU Path Tracing Tutorial 4 by Sam Lapere
///				 available at http://raytracey.blogspot.co.uk/2016/09/gpu-path-tracing-tutorial-4-optimised.html
///				 Supplementary source code is available at https://github.com/straaljager/GPU-path-tracing-tutorial-4
///
///				 Some code from the CudaTracerLib by Hannes Hergeth is also used, available at https://github.com/hhergeth/CudaTracerLib
///
/// \authors Teemu Lindborg
/// \version 1.0
/// \date 04/05/17 Updated to NCCA Coding standard
/// Revision History :
/// Initial Version 08/12/16
/// \todo -
///

#pragma once

#include "BVHNodes.h"
#include "vDataTypes.h"

///
/// \brief The SBVH class The class builds the acceleration structure when given a vector of triangles and vertices
///
class SBVH
{
private:
	///
	/// \brief The TriangleRef struct Triangle reference, only need the index of the triangle and its bounding box for the calculations
	///
	struct TriangleRef
	{
		///
		/// \brief m_triIdx Index of the triangle being referenced
		///
		unsigned int m_triIdx;

		///
		/// \brief m_bb Bounding box containing the triangle
		///
		AABB m_bb;
	};

	///
	/// \brief The NodeSpec struct Simple struct containing data used to build a node
	///
	struct NodeSpec
	{
		///
		/// \brief m_numTris Number of triangles referenced by the node
		///
		unsigned int m_numTris;

		///
		/// \brief m_bb Bounding box of the node
		///
		AABB m_bb;
	};

	///
	/// \brief The ObjectSplit struct Struct to hold the results of an object split as defined in regular BVH
	///
	struct ObjectSplit
	{
		///
		/// \brief ObjectSplit Default Ctor
		///
		ObjectSplit() :
			m_cost(FLT_MAX), m_splitAxis(-1), m_leftTris(0), m_leftSplit(AABB()), m_rightSplit(AABB()) {}

		///
		/// \brief m_cost Cost of tracing the node if object split is used
		///
		float m_cost;

		///
		/// \brief m_splitAxis Axis to split the node in
		///
		int m_splitAxis;

		///
		/// \brief m_leftTris Number of triangles for the left child
		///
		int m_leftTris;

		///
		/// \brief m_leftSplit AABB for the left child
		///
		AABB m_leftSplit;

		///
		/// \brief m_rightSplit AABB for the right child
		///
		AABB m_rightSplit;
	};

	///
	/// \brief The SpatialSplit struct Struct to hold the results of a spatial split as defined in SBVH
	///
	struct SpatialSplit
	{
		///
		/// \brief SpatialSplit Default ctor
		///
		SpatialSplit() :
			m_cost(FLT_MAX), m_splitPosition(FLT_MAX), m_axis(-1) {}

		///
		/// \brief m_cost Cost of tracing the node if spatial split is used
		///
		float m_cost;

		///
		/// \brief m_splitPosition Position to split the node at
		///
		float m_splitPosition;

		///
		/// \brief m_axis Axis to split the node in
		///
		int m_axis;
	};

	///
	/// \brief The SpatialSplitBin struct Bin structure for the spatial split as defined in SBVH
	///
	struct SpatialSplitBin
	{
		///
		/// \brief m_entries Number of triangles entering the bin (from left)
		///
		unsigned int m_entries;

		///
		/// \brief m_exits Number of triangles exiting the bin (from right)
		///
		unsigned int m_exits;

		///
		/// \brief m_bounds Bounds of the bin
		///
		AABB m_bounds;
	};

public:
	///
	/// \brief SBVH Default ctor for the SBVH, calculates the AABB for the root node and builds the triangle reference vector and
	///							starts building the acceleration structure from the root
	/// \param _triangles Triangles of the mesh
	/// \param _verts Vertices of the mesh
	/// \param _numTris Number of triangles in the mesh
	///
	SBVH(vHTriangle *_triangles, vHVert *_verts, unsigned int _numTris);

	///
	/// \brief getRoot Get the root node
	/// \return The root node
	///
	BVHNode *getRoot() const { return m_root; }

	///
	/// \brief getTriIndex Get the index of a triangle from the triangle reference stack
	/// \param _i Index to the reference stack
	/// \return Index of the triangle referenced
	///
	unsigned int getTriIndex(const unsigned int &_i) const { return m_triIndices[_i]; }

	///
	/// \brief getNodeCount Get the number of nodes in the final acceleration structure
	/// \return Number of nodes in the structure
	///
	unsigned int getNodeCount() { return m_nodes; }

private:
	///
	/// \brief buildNode Build the tree recursively using a given node specification
	/// \param _nodeSpec Node specification to use for building the node
	/// \return The generated node
	///
	BVHNode *buildNode(const NodeSpec &_nodeSpec);

	///
	/// \brief createLeaf Create a leaf node from the node spec
	/// \param _nodeSpec Node specification used to create the leaf
	/// \return A pointer to the allocated node
	///
	BVHNode *createLeaf(const NodeSpec &_nodeSpec);

	///
	/// \brief objectSplit Find the best object split for the node
	/// \param _nodeSpec Node specification used for finding the object split
	/// \param _firstTriRefIndex Index to the first triangle of the node in the triangle reference stack
	/// \param _nodeCost Cost of tracing the current node
	/// \return The best object split found
	///
	ObjectSplit objectSplit(const NodeSpec &_nodeSpec, const unsigned int &_firstTriRefIndex, const float &_nodeCost);

	///
	/// \brief spatialSplit Find the best spatial split for the node
	/// \param _nodeSpec Node specification used for finding the
	/// \param _firstTriRefIndex Index to the first triangle of the node in the triangle reference stack
	/// \param _nodeCost Cost of tracing the current node
	/// \return The best spatial split found
	///
	SpatialSplit spatialSplit(const NodeSpec &_nodeSpec, const unsigned int &_firstTriRefIndex, const float &_nodeCost);

	///
	/// \brief referenceUnsplit Unsplit a reference as explained in the SBVH paper
	/// \param o_leftSpec New node specification for the left child
	/// \param o_rightSpec New node specification for the right child
	/// \param _nodeSpec Current node specification
	/// \param _spatialSplitCandidate Details of the best spatial split found before
	///
	void referenceUnsplit(NodeSpec &o_leftSpec, NodeSpec &o_rightSpec, const NodeSpec &_nodeSpec, const SpatialSplit &_spatialSplitCandidate);

	///
	/// \brief splitReference Split a reference as explained in the SBVH paper
	/// \param o_leftSplitRef Triangle reference to left side of the split triangle
	/// \param o_rightSplitRef Triangle reference to right side of the split triangle
	/// \param _triRef Current triangle reference
	/// \param _axis Axis to split the triangle reference on
	/// \param _splitPosition Position to split the triangle on
	///
	void splitReference(TriangleRef &o_leftSplitRef, TriangleRef &o_rightSplitRef, const TriangleRef &_triRef, const int &_axis, const float &_splitPosition);

	///
	/// \brief sortTriRefStack Sort the triangle reference stack on a given axis
	/// \param _axis Axis to sort the triangle reference stack with
	/// \param _first Index of the first triangle
	/// \param _last Index of the last triangle
	///
	void sortTriRefStack(const unsigned int &_axis, const unsigned int &_first, const unsigned int &_last);

	///
	/// \brief insertionSort Simple insertion sort
	/// \param _axis Axis used for sorting
	/// \param _start
	/// \param _size
	///
	void insertionSort(const unsigned int &_axis, const unsigned int &_start, const unsigned int &_size);

	///
	/// \brief compareBounds Comparison function used to sort the triangle refs
	/// \param _axis Axis to compare on
	/// \param _a Index of the first triangle to compare
	/// \param _b Index of the second triangle to compare
	/// \return -1, 0 or 1 depending on the comparison result
	///
	int compareBounds(const unsigned int &_axis, const unsigned int &_a, const unsigned int &_b);

	///
	/// \brief m_root Root node from the process
	///
	BVHNode *m_root;

	///
	/// \brief m_triangles Pointer to the triangles of the mesh
	///
	vHTriangle *m_triangles;

	///
	/// \brief m_vertices Pointer to the vertices of the mesh
	///
	vHVert *m_vertices;

	///
	/// \brief m_triIndices Vector containing the indices of the triangles
	///
	std::vector<unsigned int> m_triIndices;

	///
	/// \brief m_triangleRefStack The triangle ref stack
	///
	std::vector<TriangleRef> m_triangleRefStack;

	///
	/// \brief m_leftBounds AABB vector, used when finding the best splits
	///
	std::vector<AABB> m_leftBounds;

	///
	/// \brief m_bins Spatial split bins when doing reference splitting
	///
	SpatialSplitBin m_bins[kSpatialBins];

	///
	/// \brief m_nodes Node count
	///
	unsigned int m_nodes;

	///
	/// \brief m_triangleCount Triangle count
	///
	unsigned int m_triangleCount;

	///
	/// \brief m_overlapThreshold Maximum overlap threshold
	///
	float m_overlapThreshold;

	///
	/// \brief m_totalObjectOverlap Total overlap in object splits, used for debugging
	///
	float m_totalObjectOverlap;

	///
	/// \brief m_totalSpatialOverlap Total overlap in spatial splits, used for debugging
	///
	float m_totalSpatialOverlap;
};
