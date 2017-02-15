/*
*  Copyright (c) 2009-2011, NVIDIA Corporation
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions are met:
*      * Redistributions of source code must retain the above copyright
*        notice, this list of conditions and the following disclaimer.
*      * Redistributions in binary form must reproduce the above copyright
*        notice, this list of conditions and the following disclaimer in the
*        documentation and/or other materials provided with the distribution.
*      * Neither the name of NVIDIA Corporation nor the
*        names of its contributors may be used to endorse or promote products
*        derived from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
*  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
*  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
*  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
*  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
*  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
*  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

///
/// @brief SBVHNode types
/// Adapted from :-
/// Sam Lapere (January 18, 2016). GPU path tracing tutorial 3: GPU-friendly Acceleration Structures. Now you're cooking with GAS! [online].
/// [Accessed 2017]. Available from: http://raytracey.blogspot.co.uk/2016/01/gpu-path-tracing-tutorial-3-take-your.html & https://github.com/straaljager/GPU-path-tracing-tutorial-3/
///
/// Original class: BVHNode
/// Reformatted the original code to match the code layout and my implementation
///
/// @author Teemu Lindborg
/// @version 0.1
/// @date 13/01/17 Initial version
/// Revision History :
/// -
/// @todo Tidying up and rewriting
///

#include <assert.h>

#include "AABB.h"
#include "Platform.h"

enum SBVH_STAT
{
	SBVH_STAT_NODE_COUNT,
	SBVH_STAT_INNER_COUNT,
	SBVH_STAT_LEAF_COUNT,
	SBVH_STAT_TRIANGLE_COUNT,
	SBVH_STAT_CHILDNODE_COUNT,
};

class SBVHNode
{
public:

	///
	/// \brief SBVHNode Default ctor
	///
	SBVHNode() :
		m_probability(1.f),
		m_parentProbability(1.f),
		m_treelet(-1),
		m_index(-1)
	{}

	virtual ~SBVHNode() {}

	///
	/// \brief isLeaf
	/// \return
	///
	virtual bool isLeaf() const = 0;

	///
	/// \brief getNumChildNodes
	/// \return
	///
	virtual unsigned int getNumChildNodes() const = 0;

	///
	/// \brief getChildNode
	/// \param _idx
	/// \return
	///
	virtual SBVHNode* getChildNode(const unsigned int &_idx) const = 0;

	///
	/// \brief getNumTriangles
	/// \return
	///
	virtual int getNumTriangles() const
	{
		return 0;
	}

	///
	/// \brief getArea
	/// \return
	///
	float getArea() const
	{
		return m_bounds.area();
	}

	///
	/// \brief getSubtreeSize Recursively counts some type of nodes (either leafnodes, innernodes, childnodes) or number of triangles
	/// \param _stat
	/// \return
	///
	int getSubtreeSize(const SBVH_STAT &_stat = SBVH_STAT_NODE_COUNT) const;

	///
	/// \brief computeSubtreeProbabilities
	/// \param _platform
	/// \param _parentProbability
	/// \param _sah
	///
	void computeSubtreeProbabilities(const Platform &_platform, const float &_parentProbability, float &_sah);

	///
	/// \brief computeSubtreeSAHCost Computes the surface area heuristics cost for the node. Assumes valid probabilities
	/// \param _platform
	/// \return
	///
	float computeSubtreeSAHCost(const Platform &_platform) const;

	///
	/// \brief deleteSubtree
	///
	void deleteSubtree();

	///
	/// \brief assignIndicesDepthFirst
	/// \param _index
	/// \param _includeLeafNodes
	///
	void assignIndicesDepthFirst(int _index = 0, const bool &_includeLeafNodes = true);

	///
	/// \brief assignIndicesBreadthFirst
	/// \param _index
	/// \param _includeLeafNodes
	///
	void assignIndicesBreadthFirst(int _index = 0, const bool &_includeLeafNodes = true);

protected:
	///
	/// \brief assignIndicesDepthFirstRecursive
	/// \param _node
	/// \param _index
	/// \param _includeLeafNodes
	///
	void assignIndicesDepthFirstRecursive(SBVHNode *_node, int &_index, const bool &_includeLeafNodes);

	///
	/// \brief m_bounds
	///
	AABB m_bounds;

	// These are somewhat experimental, for some specific test and may be invalid...
	///
	/// \brief m_probability
	///
	float m_probability; // probability of coming here (wideSBVH uses this)

	///
	/// \brief m_parentProbability
	///
	float m_parentProbability; // probability of coming to parent (wideSBVH uses this)

	///
	/// \brief m_treelet
	///
	int m_treelet; // for queuing tests (qmachine uses this)

	///
	/// \brief m_index
	///
	int m_index; // in linearized tree (qmachine uses this)
};


class InnerNode : public SBVHNode
{
public:
	///
	/// \brief InnerNode Default ctor
	/// \param _bounds
	/// \param _firstChild
	/// \param _secondChild
	///
	InnerNode(const AABB &_bounds, SBVHNode *_firstChild, SBVHNode *_secondChild)
	{
		m_bounds = _bounds;
		m_children[0] = _firstChild;
		m_children[1] = _secondChild;
	}

	///
	/// \brief isLeaf
	/// \return
	///
	bool isLeaf() const
	{
		return false;
	}

	///
	/// \brief getNumChildNodes
	/// \return
	///
	unsigned int getNumChildNodes() const
	{
		return 2;
	}

	///
	/// \brief getChildNode
	/// \param i
	/// \return
	///
	SBVHNode* getChildNode(const unsigned int &_idx) const
	{
		assert(_idx < 2);
		return m_children[_idx];
	}

private:
	///
	/// \brief m_children
	///
	SBVHNode *m_children[2];
};


class LeafNode : public SBVHNode
{
public:
	///
	/// \brief LeafNode
	/// \param _bounds
	/// \param _lowIdx
	/// \param _highIdx
	///
	LeafNode(const AABB &_bounds, const int &_lowIdx, const int &_highIdx) :
		m_lowIdx(_lowIdx),
		m_highIdx(_highIdx)
	{
		m_bounds = _bounds;
	}

	///
	/// \brief LeafNode Default copy ctor
	/// \param _s Node to copy
	///
	LeafNode(const LeafNode &_s)
	{
		*this = _s;
	}

	///
	/// \brief isLeaf
	/// \return
	///
	bool isLeaf() const
	{
		return true;
	}

	///
	/// \brief getNumChildNodes
	/// \return
	///
	unsigned int getNumChildNodes() const
	{
		return 0;
	}

	///
	/// \brief getChildNode Leaf node has no children
	/// \return nullptr
	///
	SBVHNode* getChildNode(const unsigned int &) const
	{
		return nullptr;
	}

	///
	/// \brief getNumTriangles Calculates the number of triangles based on the first and the last triangle index defined in the node
	/// \return Number of the triangles in contained by this node
	///
	int getNumTriangles() const
	{
		return m_highIdx - m_lowIdx;
	}

private:
	///
	/// \brief m_lowIdx Index of the first triangle in this node
	///
	int m_lowIdx;


	///
	/// \brief m_highIdx Index of the last triangle in this node
	///
	int m_highIdx;
};
