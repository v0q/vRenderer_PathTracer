/*
* Copyright (c) 2009-2011, NVIDIA Corporation
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* * Neither the name of NVIDIA Corporation nor the
* names of its contributors may be used to endorse or promote products
* derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

///
/// @brief Generates a SBVH tree to be used to optimise/speed up GPU path tracing
/// Modified from :-
/// Sam Lapere (September 20, 2016). GPU path tracing tutorial 4: Optimised BVH building, faster traversal and intersection kernels and HDR environment lighting [online].
/// [Accessed 2017]. Available from: http://raytracey.blogspot.co.uk/2016/09/gpu-path-tracing-tutorial-4-optimised.html & https://github.com/straaljager/GPU-path-tracing-tutorial-4
///
/// Reformatted the original code to match the code layout and my implementation
///
/// @author Teemu Lindborg
/// @version 0.1
/// @date 15/01/17 Initial version
/// Revision History :
/// -
/// @todo Tidying up and rewriting
///

#include <cfloat>
#include <vector>

#include "AABB.h"
#include "SBVHNodes.h"
#include "vDataTypes.h"

//Platform()
//{
//	m_name = std::string("Default");
//	m_SAHNodeCost = 1.f;
//	m_SAHTriangleCost = 1.f;
//	m_nodeBatchSize = 1;
//	m_triBatchSize = 1;
//	m_minLeafSize = 1;
//	m_maxLeafSize = 0x7FFFFFF;
//} /// leafsize = aantal tris

///
/// \brief The SBVH class
///
class SBVH
{
private:
	enum : unsigned int
	{
		MaxDepth = 64,
		MaxSpatialDepth = 48,
		NumSpatialBins = 32,
	};

	///
	/// \brief The Reference struct A single AABB bounding box enclosing 1 triangle, a reference can be duplicated by a split to be contained in 2 AABB boxes
	///
	struct Reference
	{
		int m_triIdx;
		AABB m_bounds;

		Reference() : m_triIdx(-1) {}
	};

	///
	/// \brief The NodeSpec struct
	///
	struct NodeSpec
	{
		unsigned int m_numRef; // number of references contained by node
		AABB m_bounds;

		NodeSpec() :
			m_numRef(0)
		{}
	};

	struct ObjectSplit
	{
		float m_sah; // cost
		int m_sortDim; // axis along which triangles are sorted
		int m_numLeft; // number of triangles (references) in left child
		AABB m_leftBounds;
		AABB m_rightBounds;

		ObjectSplit() :
			m_sah(FLT_MAX),
			m_sortDim(0),
			m_numLeft(0)
		{}
	};

	struct SpatialSplit
	{
		float m_sah;
		int m_dim; // split axis
		float m_pos; // position of split along axis (dim)

		SpatialSplit() :
			m_sah(FLT_MAX),
			m_dim(0),
			m_pos(0.0f)
		{}
	};

	struct SpatialBin
	{
		AABB m_bounds;
		int m_enter;
		int m_exit;
	};

public:
	///
	/// \brief SBVH Delete the default ctor cause we want references to the triangle and vertex vectors
	///
	SBVH() = delete;

	///
	/// \brief SBVH Default ctor, stores reference to the the given vectors to avoid unnecessary copying
	/// \param _tris Vector of the scene triangles
	/// \param _verts vector of the scene vertices
	///
	SBVH(std::vector<vHTriangle> &_tris, std::vector<vFloat3> &_verts, const float &_sA = 1.0e-5f);

	///
	/// \brief ~SBVH Default dtor
	///
	~SBVH() {}

//	CacheFriendlySBVHNode* createSBVH(const std::vector<vHVert> &_vertices, const std::vector<vHTriangle> &_triangles);
private:
	static int sortCompare(void *io_data, const unsigned int &_idxA, const unsigned int &_idxB);
	static void sortSwap(void *io_data, const unsigned int &_idxA, const unsigned int &_idxB);

	SBVHNode* exec();
	SBVHNode* buildNode(const NodeSpec &_spec, const unsigned int &_level);
	SBVHNode* createLeaf(const NodeSpec &_spec);

	ObjectSplit findObjectSplit(const NodeSpec &_spec, const float &_nodeSAH);
	void performObjectSplit(NodeSpec &_left, NodeSpec &_right, const NodeSpec &_spec, const ObjectSplit &_split);

	SpatialSplit findSpatialSplit(const NodeSpec &_spec, const float &_nodeSAH);
	void performSpatialSplit(NodeSpec &_left, NodeSpec &_right, const NodeSpec &_spec, const SpatialSplit &_split);

	void splitReference(Reference &_left, Reference &_right, const Reference &_ref, const unsigned int &_dim, const float &_pos);

	std::vector<vHTriangle> &m_triangles;
	std::vector<vFloat3> &m_vertices;

	std::vector<unsigned int> m_triIndices;
	std::vector<Reference> m_refStack;
	std::vector<AABB> m_rightBounds;

	SpatialBin m_bins[3][NumSpatialBins];

	Platform m_platform;

	unsigned int m_numDuplicates;
	unsigned int m_sortDim;
	float m_minOverlap;
	float m_splitAlpha;
	bool m_initialised;

//	CacheFriendlySBVHNode* m_cfSBVH;
};
