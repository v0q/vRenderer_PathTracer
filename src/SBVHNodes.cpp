#include "SBVHNodes.h"

float SBVHNode::surfaceArea() const
{
	return m_bounds.surfaceArea();
}

void SBVHNode::computeIntersectionProbability(const float &_probability)
{
	m_intersectionProbability = _probability;

	for(unsigned int i = 0; i < numChildNodes(); ++i)
	{
		SBVHNode *child = childNode(i);
		child->computeIntersectionProbability(_probability * (child->surfaceArea() / surfaceArea()));
	}
}

float SBVHNode::computeSAHCost() const
{
	float cost = m_intersectionProbability * (numChildNodes() * kNodeCost + numTriangles() * kTriangleCost);

	for(unsigned int i = 0; i < numChildNodes(); ++i)
	{
		cost += childNode(i)->computeSAHCost();
	}

	return cost;
}

void SBVHNode::cleanUp()
{
	for(unsigned int i = 0; i < numChildNodes(); ++i)
	{
		childNode(i)->cleanUp();
	}

	delete this;
}
