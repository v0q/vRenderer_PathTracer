///
/// \file BVHNodes.cpp
/// \brief Nodes/node methods used in the SBVH
///

#include "BVHNodes.h"

unsigned int BVHNode::nodeCount() const
{
	// Recursively counts the nodes
	unsigned int count = 1;
	for(unsigned int i = 0; i < numChildNodes(); ++i)
	{
		count += childNode(i)->nodeCount();
	}

	return count;
}

float BVHNode::surfaceArea() const
{
	// Get the surface area of the bounds
	return m_bounds.surfaceArea();
}

void BVHNode::cleanUp()
{
	// Recursively free the allocated memory
	for(unsigned int i = 0; i < numChildNodes(); ++i)
	{
		childNode(i)->cleanUp();
	}

	delete this;
}
