#include <assert.h>

#include "BVH.h"

//#define BVH_DEBUG
//#define BVH_DEBUG_LEAFS
//#define BVH_DEBUG_SPLIT
//#define BVH_DEBUG_SORT

constexpr unsigned int kQuickSortStackSize = 32;
constexpr unsigned int kQuickSortMinSize = 16;

BVH::BVH(vHTriangle *_triangles, ngl::Vec3 *_verts, unsigned int _numTris) :
	m_triangles(_triangles),
	m_vertices(_verts),
	m_triangleCount(_numTris)
{
	// Init the root node
	NodeSpec root;
	root.m_numTris = _numTris;

	// Loop through the triangles and extend the bounding box to contain each triangle
	for(unsigned int i = 0; i < _numTris; ++i)
	{
		const vHTriangle &tri = _triangles[i];

		// Create a reference to the triangle, e.g. store its index and aabb
		TriangleRef triRef;
		triRef.m_triIdx = i;
		// Loop through the verts of the triangle
		for(unsigned int j = 0; j < 3; ++j)
			triRef.m_bb.extendBB(_verts[tri.m_indices[j]]);

#ifdef BVH_DEBUG
		triRef.m_bb.printBounds();
#endif

		root.m_bb.extendBB(triRef.m_bb);
		m_triangleRefStack.push_back(triRef);
	}

	// Keeping a vector of the left bounds to avoid allocating and deallocating memory when doing recursion
	m_leftBounds.resize(root.m_numTris);

	// Build the BVH tree recursively
	m_root = buildNode(root);

#ifdef BVH_DEBUG
	root.m_bb.printBounds();
	std::cout << "Node count: " << m_root->nodeCount() << "\n";
#endif
}

BVHNode *BVH::buildNode(const NodeSpec &_nodeSpec)
{
	// No reason to perform recursion if we've reached the minimum allowed triangles per node
	if(_nodeSpec.m_numTris <= kMinLeafSize)
	{
		return createLeaf(_nodeSpec);
	}
	// Algorithm 1. Centroid-based SAH partitioning
	float leafCost = _nodeSpec.m_bb.surfaceArea() * kTriangleCost * _nodeSpec.m_numTris;
	int splitAxis = -1;
	int leftTris = -1;
	int firstTriRefIndex = m_triangleRefStack.size() - _nodeSpec.m_numTris;

	AABB leftSplit;
	AABB rightSplit;

#ifdef BVH_DEBUG
	#ifdef BVH_DEBUG_SPLIT
		std::cout << "Initial leaf cost: " << leafCost << "\n";
	#endif
#endif

	// Loop through the axes: 0 = X, 1 = Y, 2 = Z
	for(unsigned int axis = 0; axis < 3; ++axis)
	{
		// Sort triangle list using centroid of their aabb in current axis
		sortTriRefStack(axis, firstTriRefIndex, m_triangleRefStack.size());

		// Sweep from left
		AABB leftBound;
		for(unsigned int i = 0; i < _nodeSpec.m_numTris; ++i)
		{
			/*
			 * S[i].leftArea = Area(S1) {with Area(Empty) = ∞}
			 * move triangle i from S2 to S1
			 */
			leftBound.extendBB(m_triangleRefStack[firstTriRefIndex + i].m_bb);
			m_leftBounds[firstTriRefIndex + i] = leftBound;
		}

		// Sweep from right
		AABB rightBound;
		for(int i = _nodeSpec.m_numTris; i > 0; --i)
		{
			rightBound.extendBB(m_triangleRefStack[firstTriRefIndex + (i - 1)].m_bb);
			// Evaluate Equation 2
			// T = 2TAABB + A(S1)/A(S) *	N(S1)Ttri + A(S2)/A(S) * N(S2)Ttri
			// f(b) = LSA(b) · L(b) + RSA(b)·(n -L(b) - SA · n
			float cost = m_leftBounds[firstTriRefIndex + (i - 1)].surfaceArea() * kTriangleCost * i + rightBound.surfaceArea() * (_nodeSpec.m_numTris - (i - 1));
			// move Triangle i from S1 to S2 is done by storing all of the left bounds to a vector
			if(cost < leafCost)
			{
				leafCost = cost;
				leftTris = i;
				splitAxis = axis;
				leftSplit = m_leftBounds[firstTriRefIndex + (i - 1)];
				rightSplit = rightBound;
			}
		}
	}

#ifdef BVH_DEBUG
	#ifdef BVH_DEBUG_SPLIT
		std::cout << "Leaf cost: " << leafCost << '\n';
		std::cout << "Left node tris: " << leftTris << '\n';
		std::cout << "Best split axis: " << (splitAxis < 0 ? "Not found" : (splitAxis == 0 ? "X" : (splitAxis == 1 ? "Y" : "Z"))) << '\n';
	#endif
#endif

	if(splitAxis == -1)
	{
		return createLeaf(_nodeSpec);
	}
	else
	{
		sortTriRefStack(splitAxis, firstTriRefIndex, m_triangleRefStack.size());
		NodeSpec leftSpec;
		leftSpec.m_bb = leftSplit;
		leftSpec.m_numTris = leftTris;

		NodeSpec rightSpec;
		rightSpec.m_bb = rightSplit;
		rightSpec.m_numTris = _nodeSpec.m_numTris - leftTris;

		BVHNode *rightChild = buildNode(rightSpec);
		BVHNode *leftChild = buildNode(leftSpec);

		return new InnerNode(_nodeSpec.m_bb, leftChild, rightChild);
	}
}

BVHNode* BVH::createLeaf(const NodeSpec &_nodeSpec)
{
	for(unsigned int i = 0; i < _nodeSpec.m_numTris; ++i)
	{
		m_triIndices.push_back(m_triangleRefStack.back().m_triIdx);
		m_triangleRefStack.pop_back();
	}

	LeafNode *leaf = new LeafNode(_nodeSpec.m_bb, m_triIndices.size() - _nodeSpec.m_numTris, m_triIndices.size());

#ifdef BVH_DEBUG
	#ifdef BVH_DEBUG_LEAFS
		std::cout << "Created a leaf node\n";
		std::cout << "  Leaf node:\n";
		std::cout << "    Num tris: " << leaf->numTriangles() << "\n";
		std::cout << "    Indices: [" << leaf->firstIndex() << "-" << leaf->lastIndex() << "]\n";
		_nodeSpec.m_bb.printBounds();
	#endif
#endif

	return leaf;
}

void BVH::sortTriRefStack(const unsigned int &_axis, const unsigned int &_first, const unsigned int &_last)
{
#ifdef BVH_DEBUG
	#ifdef BVH_DEBUG_SORT
		std::cout << "Triangle refs before sorting along " << (_axis == 0 ? 'X' : (_axis == 1 ? 'Y' : 'Z')) << "-axis:\n";
		for(unsigned int i = _first; i < _last; ++i)
		{
			const TriangleRef &triRef = m_triangleRefStack[i];
			std::cout << "  Idx: " << triRef.m_triIdx << ", Centroid: " << (triRef.m_bb.minBounds().m_openGL[_axis] + triRef.m_bb.maxBounds().m_openGL[_axis])/2.f << "\n";
		}
	#endif
#endif
	/// The following section is adapted from :-
	/// XXX [online].
	/// [Accessed 2017]. Available from: https://github.com/hhergeth/CudaTracerLib/blob/master/Engine/SceneBuilder/SplitBVHBuilder.cpp.
	int stack[kQuickSortStackSize];
	int sp = 0;
	int low = _first;
	int high = _last;

	stack[sp++] = high;

	while(sp)
	{
		high = stack[--sp];
		assert(low <= high);

		// Small enough or stack full => use insertion sort.
		if(high - low < static_cast<int>(kQuickSortMinSize) || sp + 2 > static_cast<int>(kQuickSortStackSize))
		{
			insertionSort(_axis, low, high - low);
			low = high + 1;
			continue;
		}

		// Swap pivot to the highest point
		vUtilities::swap(m_triangleRefStack[(low + high)/2], m_triangleRefStack[high - 1]);

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

			vUtilities::swap(m_triangleRefStack[i], m_triangleRefStack[j]);
		}

		// Swap pivot back
		vUtilities::swap(m_triangleRefStack[i], m_triangleRefStack[high - 1]);

		assert(sp + 2 <= static_cast<int>(kQuickSortStackSize));
		if(high - i > 2)
			stack[sp++] = high;
		if(i - low > 1)
			stack[sp++] = i;
		else
			low = i + 1;
	}
#ifdef BVH_DEBUG
	#ifdef BVH_DEBUG_SORT
		std::cout << "Triangle refs after sorting along " << (_axis == 0 ? 'X' : (_axis == 1 ? 'Y' : 'Z')) << "-axis:\n";
		for(unsigned int i = _first; i < _last; ++i)
		{
			const TriangleRef &triRef = m_triangleRefStack[i];
			std::cout << "  Idx: " << triRef.m_triIdx << ", Centroid: " << (triRef.m_bb.minBounds().m_openGL[_axis] + triRef.m_bb.maxBounds().m_openGL[_axis])/2.f << "\n";
		}
		std::cout << "\n";
	#endif
#endif
}

void BVH::insertionSort(const unsigned int &_axis, const unsigned int &_start, const unsigned int &_size)
{
	/// The following section is adapted from :-
	/// XXX [online].
	/// [Accessed 2017]. Available from: https://github.com/hhergeth/CudaTracerLib/blob/master/Engine/SceneBuilder/SplitBVHBuilder.cpp.
	for(unsigned int i = 1; i < _size; ++i)
	{
		int j = _start + i - 1;
		while(j >= static_cast<int>(_start) && compareBounds(_axis, j + 1, j) < 0)
		{
			vUtilities::swap(m_triangleRefStack[j], m_triangleRefStack[j + 1]);
			--j;
		}
	}
}

int BVH::compareBounds(const unsigned int &_axis, const unsigned int &_a, const unsigned int &_b)
{
	// Compare AABB per-axis centroids (used for sorting the triangles based on their centroids on an axis)
	const TriangleRef &triA = m_triangleRefStack[_a];
	const TriangleRef &triB = m_triangleRefStack[_b];

	// No need to divide for the centroid as if minA + maxA > minB + minB, then (minA + maxA)/2 > (minB + maxB)/2
	float centroidA = triA.m_bb.minBounds().m_openGL[_axis] + triA.m_bb.maxBounds().m_openGL[_axis];
	float centroidB = triB.m_bb.minBounds().m_openGL[_axis] + triB.m_bb.maxBounds().m_openGL[_axis];

	return (centroidA < centroidB ? -1 :
					(centroidA > centroidB) ? 1 :
					// If centroids were the same, ordering based on triangle indices
					 (triA.m_triIdx < triB.m_triIdx) ? -1 :
						(triA.m_triIdx > triB.m_triIdx) ? 1 : 0);
}
