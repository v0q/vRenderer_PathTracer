#include <iostream>
#include "BVHNodes.h"

unsigned int BVHNode::nodeCount() const
{
	unsigned int count = 1;
	for(unsigned int i = 0; i < numChildNodes(); ++i)
	{
		count += childNode(i)->nodeCount();
	}

	return count;
}

float BVHNode::surfaceArea() const
{
	return m_bounds.surfaceArea();
}

void BVHNode::cleanUp()
{
	for(unsigned int i = 0; i < numChildNodes(); ++i)
	{
		childNode(i)->cleanUp();
	}

	delete this;
}
