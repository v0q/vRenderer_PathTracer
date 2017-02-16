#include <assert.h>
#include <iostream>

#include "SBVH.h"

#define SBVH_DEBUG_ON

void SBVH::exec()
{
	assert(m_initialised);

	NodeSpec rootSpecification;
	rootSpecification.m_numRef = m_triangles.size();

	m_triRefStack.resize(m_triangles.size());

	for(unsigned int i = 0; i < m_triangles.size(); ++i)
	{
		m_triRefStack[i].m_triIndex = i;

		for(unsigned int j = 0; j < 3; ++j)
			m_triRefStack[i].m_bounds.extendBB(m_vertices[m_triangles[i].m_indices[j]]);

		rootSpecification.m_bounds.extendBB(m_triRefStack[i].m_bounds);
	}

#ifdef SBVH_DEBUG_ON
	std::cout << "Root bounds:\n";
	std::cout << "  Min: " << rootSpecification.m_bounds.minBounds().m_x << ", " << rootSpecification.m_bounds.minBounds().m_y << ", " << rootSpecification.m_bounds.minBounds().m_z << "\n";
	std::cout << "  Max: " << rootSpecification.m_bounds.maxBounds().m_x << ", " << rootSpecification.m_bounds.maxBounds().m_y << ", " << rootSpecification.m_bounds.maxBounds().m_z << "\n";
#endif

	m_root = buildSBVH(rootSpecification, 0);
}

SBVHNode* SBVH::buildSBVH(const NodeSpec &_nSpec, const unsigned int &_depth)
{
	// Check if we've reached depth or leaf size limit and create a leaf node
	if(_nSpec.m_numRef <= kMinLeafSize || _depth >= kMaxDepth)
	{
		return createLeafNode(_nSpec);
	}

	// Find candidates for splitting
	float surfaceArea = _nSpec.m_bounds.surfaceArea();
	// Estimate the costs to decide whether we should create a node or a leaf
	float leafCost = surfaceArea * (_nSpec.m_numRef * kTriangleCost);
	float nodeCost = surfaceArea * (2 * kNodeCost);

	// SBVH Algorithm
	// 1. Find an object split candidate
	findObjectSplitCandidate(_nSpec, nodeCost);
	exit(0);

	SBVHNode *leftChild;
	SBVHNode *rightChild;

	return new InnerNode(_nSpec.m_bounds, leftChild, rightChild);
}

SBVHNode* SBVH::createLeafNode(const NodeSpec &_nSpec)
{
	return new LeafNode(_nSpec.m_bounds, m_triIndices.size() - _nSpec.m_numRef, m_triIndices.size());
}

void SBVH::findObjectSplitCandidate(const NodeSpec &_nSpec, const float &_nodeCost)
{
	float sahCost = FLT_MAX;
	int bestAxis = -1;
	unsigned int firstRefIndex = m_triRefStack.size() - _nSpec.m_numRef;
	// Loop through all the axes, X = 0, Y = 1, Z = 2
	for(unsigned int axis = 0; axis < 3; ++axis)
	{
		// Following the Algorithm 1 described in http://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/wald07_packetbvh.pdf

		/*
		 * In each partitioning step, the primitive list of the current node P
		 * is sorted based on the centroids of the primitive AABBs.
		 */
		sortTriRefStack(axis, m_triRefStack.size() - _nSpec.m_numRef, m_triRefStack.size(), true);

		/*
		 * Sweep from left
		 */
		AABB lBounds;
		for(unsigned int i = 0; i < _nSpec.m_numRef; ++i)
		{
			lBounds.extendBB(m_triRefStack[firstRefIndex + i].m_bounds);
//			m_rightBounds[i - 1] = rightBounds;
		}

		/*
		 * Sweep from right
		 */
		AABB rBounds;
//		for(int i = _nSpec.m_numRef - 1; i > 0; -ii)
//		{
//			rBounds.extendBB(m_triRefStack[i - firstRefIndex].m_bounds);
//			float sah = _nodeCost + rBounds.surfaceArea() * kTriangleCost * i + m_rightBounds[i - 1].area() * m_platform.getTriangleCost(spec.numRef - i);
//			if (sah < split.sah)
//			{
//				split.sah = sah;
//				split.sortDim = m_sortDim;
//				split.numLeft = i;
//				split.leftBounds = leftBounds;
//				split.rightBounds = m_rightBounds[i - 1];
//			}
//		}
	}
}

void SBVH::sortTriRefStack(const unsigned int &_axis, const unsigned int &_first, const unsigned int &_last, bool orig)
{
#ifdef SBVH_DEBUG_ON
	if(_first == 0 && _last == m_triRefStack.size() && orig)
	{
		std::cout << "Triangle indices before sorting by axis " << _axis << " centroids:\n  ";
		for(unsigned int k = 0; k < m_triRefStack.size(); ++k)
		{
			const TriRef &triA = m_triRefStack[k];
			std::cout << "[i: " << triA.m_triIndex << ", c: " << triA.m_bounds.minBounds().m_openGL[_axis] + triA.m_bounds.maxBounds().m_openGL[_axis] << "] ";
		}
		std::cout << "\n";
	}
#endif

	/// The following section is adapted from :-
	/// Sam Lapere (2016). GPU-path-tracing-tutorial-4 [online].
	/// [Accessed 2017]. Available from: https://github.com/straaljager/GPU-path-tracing-tutorial-4/blob/master/Sort.cpp.
	int stack[32];
	int sp = 0;
	int low = _first;
	int high = _last;
	stack[sp++] = high;

	// Quick sort
	while(sp)
	{
		high = stack[--sp];
		// Swap pivot to the highest point
		vUtilities::swap(m_triRefStack[(low + high)/2], m_triRefStack[high - 1]);

		int i = low - 1;
		int j = high - 1;
		while(1)
		{
			do
			{
				i++;
			} while(compareBounds(_axis, i, high - 1) < 0 && i < static_cast<int>(_last));
			do
			{
				j--;
			} while(compareBounds(_axis, j, high - 1) > 0 && j > static_cast<int>(_first));

			if(i >= j)
				break;

			vUtilities::swap(m_triRefStack[i], m_triRefStack[j]);
		}

		// Swap pivot back
		vUtilities::swap(m_triRefStack[i], m_triRefStack[high - 1]);

		if(high - i > 2)
			stack[sp++] = high;
		if (i - low > 1)
			stack[sp++] = i;
		else
			low = i + 1;
	}

#ifdef SBVH_DEBUG_ON
	if(_first == 0 && _last == m_triRefStack.size() && orig)
	{
		std::cout << "Triangle indices after sorting by axis " << _axis << " centroids:\n  ";
		for(unsigned int k = 0; k < m_triRefStack.size(); ++k)
		{
			const TriRef &triA = m_triRefStack[k];
			std::cout << "[i: " << triA.m_triIndex << ", c: " << triA.m_bounds.minBounds().m_openGL[_axis] + triA.m_bounds.maxBounds().m_openGL[_axis] << "] ";
		}
		std::cout << "\n";
	}
#endif
}

int SBVH::compareBounds(const unsigned int &_axis, const unsigned int &_i, const unsigned int &_j)
{
	const TriRef &triA = m_triRefStack[_i];
	const TriRef &triB = m_triRefStack[_j];

	// No need to divide for the centroid as if minA + maxA > minB + minB, then (minA + maxA)/2 > (minB + maxB)/2
	float boundsA = triA.m_bounds.minBounds().m_openGL[_axis] + triA.m_bounds.maxBounds().m_openGL[_axis];
	float boundsB = triB.m_bounds.minBounds().m_openGL[_axis] + triB.m_bounds.maxBounds().m_openGL[_axis];

	return (boundsA < boundsB ? -1 :
					(boundsA > boundsB) ? 1 :
					// If bounds were the same, ordering based on triangle indices
					 (triA.m_triIndex < triB.m_triIndex) ? -1 :
						(triA.m_triIndex > triB.m_triIndex) ? 1 : 0);
}
