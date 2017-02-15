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

#include <assert.h>

#include "SBVHNodes.h"

int SBVHNode::getSubtreeSize(const SBVH_STAT &_stat) const
{
	int count;

	switch(_stat)
	{
		default: assert(0); // unknown mode

		// counts all nodes including leafnodes
		case SBVH_STAT_NODE_COUNT:
			count = 1;
		break;

		// counts only leafnodes
		case SBVH_STAT_LEAF_COUNT:
			count = isLeaf() ? 1 : 0;
		break;

		// counts only innernodes
		case SBVH_STAT_INNER_COUNT:
			count = isLeaf() ? 0 : 1;
		break;

		// counts all triangles
		case SBVH_STAT_TRIANGLE_COUNT:
			count = isLeaf() ? reinterpret_cast<const LeafNode*>(this)->getNumTriangles() : 0;
		break;

		//counts only childnodes
		case SBVH_STAT_CHILDNODE_COUNT:
			count = getNumChildNodes();
		break;
	}

	// if current node is not a leaf node, continue counting its childnodes recursively
	if(!isLeaf())
	{
		for(unsigned int i = 0; i < getNumChildNodes(); ++i)
			count += getChildNode(i)->getSubtreeSize(_stat);
	}

	return count;
}

void SBVHNode::deleteSubtree()
{
	for(unsigned int i = 0; i < getNumChildNodes(); ++i)
		getChildNode(i)->deleteSubtree();

	delete this;
}

void SBVHNode::computeSubtreeProbabilities(const Platform &_platform, const float &_parentProbability, float &_sah)
{
	_sah += _parentProbability * _platform.getCost(this->getNumChildNodes(), this->getNumTriangles());

	m_probability = _parentProbability;

	// recursively compute probabilities and add to SAH
	for(unsigned int i = 0; i < getNumChildNodes(); ++i)
	{
		SBVHNode* child = getChildNode(i);
		/// childnode area / parentnode area
		child->m_parentProbability = _parentProbability;
		child->computeSubtreeProbabilities(_platform, _parentProbability * child->m_bounds.area() / this->m_bounds.area(), _sah);
	}
}


// TODO: requires valid probabilities...
float SBVHNode::computeSubtreeSAHCost(const Platform &_platform) const
{
	float sah = m_probability * _platform.getCost(getNumChildNodes(), getNumTriangles());

	for(unsigned int i = 0; i < getNumChildNodes(); ++i)
		sah += getChildNode(i)->computeSubtreeSAHCost(_platform);

	return sah;
}

void SBVHNode::assignIndicesDepthFirstRecursive(SBVHNode *_node, int &_index, const bool &_includeLeafNodes)
{
	if(_node->isLeaf() && !_includeLeafNodes)
		return;

	_node->m_index = _index++;

	for(unsigned int i = 0; i < getNumChildNodes(); ++i)
		assignIndicesDepthFirstRecursive(_node->getChildNode(i), _index, _includeLeafNodes);
}

void SBVHNode::assignIndicesDepthFirst(int _index, const bool &_includeLeafNodes)
{
	assignIndicesDepthFirstRecursive(this, _index, _includeLeafNodes);
}

void SBVHNode::assignIndicesBreadthFirst(int _index, const bool &_includeLeafNodes)
{
	std::vector<SBVHNode*> nodes;
	nodes.push_back(this);
	unsigned int head = 0;

	while(head < nodes.size())
	{
		// pop
		SBVHNode* node = nodes[head++];

		// discard
		if(node->isLeaf() && !_includeLeafNodes)
			continue;

		// assign
		node->m_index = _index++;

		// push children
		for(unsigned int i = 0; i < getNumChildNodes(); ++i)
			nodes.push_back(node->getChildNode(i));
	}
}
