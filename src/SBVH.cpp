#include <assert.h>

#include "SBVH.h"

#define BVH_DEBUG
#define BVH_DEBUG_SPATIAL_SPLITS
//#define BVH_DEBUG_LEAFS
//#define BVH_DEBUG_SPLIT
//#define BVH_DEBUG_SORT

constexpr float kObjectSplitAlpha = 0.0003f;
constexpr unsigned int kQuickSortStackSize = 32;
constexpr unsigned int kQuickSortMinSize = 16;

SBVH::SBVH(vHTriangle *_triangles, ngl::Vec3 *_verts, unsigned int _numTris) :
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

		root.m_bb.extendBB(triRef.m_bb);
		m_triangleRefStack.push_back(triRef);
	}

	// Calculate the min overlap SA for object split to consider spatial split
	m_overlapThreshold = root.m_bb.surfaceArea() * kObjectSplitAlpha;

	// Keeping a vector of the left bounds to avoid allocating and deallocating memory when doing recursion
	m_leftBounds.resize(std::max(root.m_numTris, kSpatialBins));

	m_totalObjectOverlap = m_totalSpatialOverlap = 0.f;

	// Build the BVH tree recursively
	m_root = buildNode(root);

#ifdef BVH_DEBUG
	std::cout << "Total object split overlap SA: " << m_totalObjectOverlap << "\n";
	std::cout << "Total spatial split overlap SA: " << m_totalSpatialOverlap << "\n";
	root.m_bb.printBounds();
	std::cout << "Node count: " << m_root->nodeCount() << "\n";
#endif
}

BVHNode *SBVH::buildNode(const NodeSpec &_nodeSpec)
{
	// No reason to perform recursion if we've reached the minimum allowed triangles per node
	if(_nodeSpec.m_numTris <= kMinLeafSize)
	{
		return createLeaf(_nodeSpec);
	}

  unsigned int firstTriRefIndex = m_triangleRefStack.size() - _nodeSpec.m_numTris;
	float surfaceArea = _nodeSpec.m_bb.surfaceArea();
	float leafCost = surfaceArea * kTriangleCost * _nodeSpec.m_numTris;
  float nodeCost = surfaceArea * kNodeCost * 2;

#ifdef BVH_DEBUG
	#ifdef BVH_DEBUG_SPLIT
		std::cout << "Initial leaf cost: " << leafCost << "\n";
	#endif
#endif

	/*
	 * 1. Find an object split candidate. This step is equivalent to the
	 *		conventional split search for a BVH node (see Section 2.1).
	 *		We use a full SAH search in our implementation, but other
	 *		variants could easily be used.
	 */
	ObjectSplit objectSplitCandidate = objectSplit(_nodeSpec, firstTriRefIndex, nodeCost);
	SpatialSplit spatialSplitCandidate;

	/*
	 * Check whether doing spatial splits would be worth it. E.g. if the object split nodes
	 * do not overlap much, there's no reason to perform/look at spatial splitting
	 * λ = SA(B1 ∩ B2)
	 *
	 * λ > α * SA(Broot)
	 */

	AABB overlap = objectSplitCandidate.m_leftSplit;
	overlap.intersectBB(objectSplitCandidate.m_rightSplit);

	m_totalObjectOverlap += overlap.surfaceArea();

	if(overlap.surfaceArea() > m_overlapThreshold)
	{
		spatialSplitCandidate = spatialSplit(_nodeSpec, firstTriRefIndex, nodeCost);
	}

	if(leafCost < objectSplitCandidate.m_cost && leafCost < spatialSplitCandidate.m_cost)
	{
		return createLeaf(_nodeSpec);
	}

	NodeSpec leftSpec;
	NodeSpec rightSpec;
	if(spatialSplitCandidate.m_cost < objectSplitCandidate.m_cost)
	{
		referenceUnsplit(leftSpec, rightSpec, _nodeSpec, spatialSplitCandidate);
	}
	else
	{
		if(objectSplitCandidate.m_splitAxis == -1)
		{
			return createLeaf(_nodeSpec);
		}
		else
		{
			sortTriRefStack(objectSplitCandidate.m_splitAxis, firstTriRefIndex, m_triangleRefStack.size());
			leftSpec.m_bb = objectSplitCandidate.m_leftSplit;
			leftSpec.m_numTris = objectSplitCandidate.m_leftTris;
			rightSpec.m_bb = objectSplitCandidate.m_rightSplit;
			rightSpec.m_numTris = _nodeSpec.m_numTris - objectSplitCandidate.m_leftTris;
		}
	}

	BVHNode *rightChild = buildNode(rightSpec);
	BVHNode *leftChild = buildNode(leftSpec);

	return new InnerNode(_nodeSpec.m_bb, leftChild, rightChild);
}

BVHNode* SBVH::createLeaf(const NodeSpec &_nodeSpec)
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

SBVH::ObjectSplit SBVH::objectSplit(const NodeSpec &_nodeSpec, const unsigned int &_firstTriRefIndex, const float &_nodeCost)
{
	// Algorithm 1. Centroid-based SAH partitioning
	ObjectSplit candidate;
	candidate.m_cost = _nodeSpec.m_bb.surfaceArea() * kTriangleCost * _nodeSpec.m_numTris;
	candidate.m_splitAxis = -1;
	candidate.m_leftTris = -1;
	candidate.m_leftSplit = AABB();
	candidate.m_rightSplit = AABB();

	// Loop through the axes: 0 = X, 1 = Y, 2 = Z
	for(unsigned int axis = 0; axis < 3; ++axis)
	{
		// Sort triangle list using centroid of their aabb in current axis
		sortTriRefStack(axis, _firstTriRefIndex, m_triangleRefStack.size());

		// Sweep from left
		AABB leftBound;
		for(size_t i = 0; i < _nodeSpec.m_numTris; ++i)
		{
			/*
			 * S[i].leftArea = Area(S1) {with Area(Empty) = ∞}
			 * move triangle i from S2 to S1
			 */
			leftBound.extendBB(m_triangleRefStack[_firstTriRefIndex + i].m_bb);
			m_leftBounds[_firstTriRefIndex + i] = leftBound;
		}

		// Sweep from right
		AABB rightBound;
		for(size_t i = _nodeSpec.m_numTris; i > 0; --i)
		{
			rightBound.extendBB(m_triangleRefStack[_firstTriRefIndex + (i - 1)].m_bb);
			// Evaluate Equation 2
			// T = 2TAABB + A(S1)/A(S) *	N(S1)Ttri + A(S2)/A(S) * N(S2)Ttri
			// f(b) = LSA(b) · L(b) + RSA(b)·(n -L(b) - SA · n
			float cost = _nodeCost + m_leftBounds[_firstTriRefIndex + (i - 1)].surfaceArea() * kTriangleCost * i + rightBound.surfaceArea() * (_nodeSpec.m_numTris - (i - 1));
			// move Triangle i from S1 to S2 is done by storing all of the left bounds to a vector
			if(cost < candidate.m_cost)
			{
				candidate.m_cost = cost;
				candidate.m_leftTris = i;
				candidate.m_splitAxis = axis;
				candidate.m_leftSplit = m_leftBounds[_firstTriRefIndex + (i - 1)];
				candidate.m_rightSplit = rightBound;
			}
		}
	}

	return candidate;
}

///
/// This section follows the paper http://www.sci.utah.edu/~wald/Publications/2007/ParallelBVHBuild/fastbuild.pdf
/// and the implementation is partly adapted from https://github.com/straaljager/GPU-path-tracing-tutorial-4/blob/master/SplitBVHBuilder.cpp
///
SBVH::SpatialSplit SBVH::spatialSplit(const NodeSpec &_nodeSpec, const unsigned int &_firstTriRefIndex, const float &_nodeCost)
{
	SpatialSplit candidate;
	candidate.m_cost = FLT_MAX;
	candidate.m_splitPosition = FLT_MAX;
	candidate.m_axis = -1;

	float axisW[3] = {_nodeSpec.m_bb.maxBounds().m_x - _nodeSpec.m_bb.minBounds().m_x,
										_nodeSpec.m_bb.maxBounds().m_y - _nodeSpec.m_bb.minBounds().m_y,
										_nodeSpec.m_bb.maxBounds().m_z - _nodeSpec.m_bb.minBounds().m_z};

	candidate.m_axis = axisW[0] > axisW[1] && axisW[0] > axisW[2] ? 0 :
										 (axisW[1] > axisW[2] && axisW[1] > axisW[0] ? 1 : 2);

	float axisOrig = _nodeSpec.m_bb.minBounds().m_openGL[candidate.m_axis];
	float binWidth = axisW[candidate.m_axis] / kSpatialBins;
	float binInvWidth = 1.f / binWidth;

	// Create and divide the spatial bins to equal width
	for(size_t i = 0; i < kSpatialBins; ++i)
	{
		m_bins[i].m_entries = 0;
		m_bins[i].m_exits = 0;
		m_bins[i].m_bounds = AABB();
		m_bins[i].m_bounds.setMinBoundsComponent(candidate.m_axis, axisOrig + binWidth*i);
		m_bins[i].m_bounds.setMaxBoundsComponent(candidate.m_axis, axisOrig + binWidth*(i+1));
	}

	for(size_t i = 0; i < _nodeSpec.m_numTris; ++i)
	{
		const TriangleRef &tri = m_triangleRefStack[_firstTriRefIndex + i];
		TriangleRef triRef = tri;
		unsigned int firstBinId = vUtilities::clamp(binInvWidth * (triRef.m_bb.minBounds().m_openGL[candidate.m_axis] - axisOrig), 0, kSpatialBins - 1);
		unsigned int lastBinId = vUtilities::clamp(binInvWidth * (triRef.m_bb.maxBounds().m_openGL[candidate.m_axis] - axisOrig), firstBinId, kSpatialBins - 1);

		for(unsigned int j = firstBinId; j < lastBinId; ++j)
		{
			// Location of the split plane (e.g. bin wall) inside our bounding box
			float splitPosition = axisOrig + binWidth*(j + 1);

			// Find if the reference needs to be split
			TriangleRef leftSplitRef;
			TriangleRef rightSplitRef;

			splitReference(leftSplitRef, rightSplitRef, triRef, candidate.m_axis, splitPosition);

			m_bins[j].m_bounds.extendBB(leftSplitRef.m_bb);
			triRef = rightSplitRef;
		}
		m_bins[lastBinId].m_bounds.extendBB(triRef.m_bb);
		m_bins[firstBinId].m_entries++;
		m_bins[lastBinId].m_exits++;
	}

	int totEntries = 0;
	int totExits = 0;
	for(size_t i = 0; i < kSpatialBins; ++i)
	{
		totEntries += m_bins[i].m_entries;
		totExits += m_bins[i].m_exits;
	}

	// Sweep from left
	AABB leftBound;
	for(size_t i = 0; i < kSpatialBins; ++i)
	{
		leftBound.extendBB(m_bins[i].m_bounds);
		m_leftBounds[i] = leftBound;
	}

	// Sweep from right
	AABB rightBound;

	// Initialise all tris to left
	int leftTris = _nodeSpec.m_numTris;
	int rightTris = 0;
	float overlap = 0.f;

	for(size_t i = kSpatialBins; i > 0; --i)
	{
		rightBound.extendBB(m_bins[i - 1].m_bounds);

		// Move tri refs from left to right
		leftTris -= m_bins[i - 1].m_entries;
		rightTris += m_bins[i - 1].m_exits;

		float cost = _nodeCost + m_leftBounds[i - 1].surfaceArea() * kTriangleCost * leftTris + rightBound.surfaceArea() * rightTris;
		if(cost < candidate.m_cost)
		{
			candidate.m_cost = cost;
			candidate.m_splitPosition = axisOrig + binWidth * i;
		}
	}

	m_totalSpatialOverlap += overlap;

//#ifdef BVH_DEBUG
//	#ifdef BVH_DEBUG_SPATIAL_SPLITS
//		std::cout << "Best spatial split candidate:\n";
//		std::cout << "  Axis: " << (candidate.m_axis == 0 ? 'X' : (candidate.m_axis == 1 ? 'Y' : 'Z')) << "\n";
//		std::cout << "  Cost: " << candidate.m_cost << "\n";
//		std::cout << "  Split pos: " << candidate.m_splitPosition << "\n";
//	#endif
//#endif

	return candidate;
}

void SBVH::referenceUnsplit(NodeSpec &o_leftSpec, NodeSpec &o_rightSpec, const NodeSpec &_nodeSpec, const SpatialSplit &_spatialSplitCandidate)
{
	int leftStart = m_triangleRefStack.size() - _nodeSpec.m_numTris;
	int leftEnd = leftStart;
	int rightStart = m_triangleRefStack.size();

	o_leftSpec.m_bb = o_rightSpec.m_bb = AABB();

	for(int i = leftEnd; i < rightStart; ++i)
	{
		const TriangleRef &tri = m_triangleRefStack[i];
		if(tri.m_bb.maxBounds().m_openGL[_spatialSplitCandidate.m_axis] <= _spatialSplitCandidate.m_splitPosition)
		{
			o_leftSpec.m_bb.extendBB(tri.m_bb);
			vUtilities::swap(m_triangleRefStack[i], m_triangleRefStack[leftEnd++]);
		}
		else if(tri.m_bb.minBounds().m_openGL[_spatialSplitCandidate.m_axis] >= _spatialSplitCandidate.m_splitPosition)
		{
			o_rightSpec.m_bb.extendBB(tri.m_bb);
			vUtilities::swap(m_triangleRefStack[i--], m_triangleRefStack[--rightStart]);
		}
	}

	while(leftEnd < rightStart)
	{
		TriangleRef leftRef;
		TriangleRef rightRef;

		splitReference(leftRef, rightRef, m_triangleRefStack[leftEnd], _spatialSplitCandidate.m_axis, _spatialSplitCandidate.m_splitPosition);

		AABB unsplitLeft = o_leftSpec.m_bb;
		AABB unsplitRight = o_rightSpec.m_bb;
		AABB duplicateLeft = o_leftSpec.m_bb;
		AABB duplicateRight = o_rightSpec.m_bb;

		unsplitLeft.extendBB(m_triangleRefStack[leftEnd].m_bb);
		unsplitRight.extendBB(m_triangleRefStack[leftEnd].m_bb);

		duplicateLeft.extendBB(leftRef.m_bb);
		duplicateRight.extendBB(rightRef.m_bb);

		float lac = kTriangleCost * (leftEnd - leftStart);
		float rac = kTriangleCost * (m_triangleRefStack.size() - rightStart);
		float lbc = kTriangleCost * (leftEnd - leftStart + 1);
		float rbc = kTriangleCost * (leftEnd - leftStart - rightStart + 1);

		float unsplitLeftCost = unsplitLeft.surfaceArea() * lbc + o_rightSpec.m_bb.surfaceArea() * rac;
		float unsplitRightCost = unsplitRight.surfaceArea() * rbc + o_leftSpec.m_bb.surfaceArea() * lac;
		float duplicateCost = duplicateLeft.surfaceArea() * lbc + duplicateRight.surfaceArea() * rbc;

		float minCost = unsplitLeftCost < unsplitRightCost ? (unsplitLeftCost < duplicateCost ? unsplitLeftCost : duplicateCost) : (unsplitRightCost < duplicateCost ? unsplitRightCost : duplicateCost);

		if(minCost == unsplitLeftCost)
		{
			o_leftSpec.m_bb = unsplitLeft;
			leftEnd++;
		}
		else if(minCost == unsplitRightCost)
		{
			o_rightSpec.m_bb = unsplitRight;
			vUtilities::swap(m_triangleRefStack[leftEnd], m_triangleRefStack[--rightStart]);
		}
		else
		{
			o_leftSpec.m_bb = duplicateLeft;
			o_rightSpec.m_bb = duplicateRight;
			m_triangleRefStack[leftEnd++] = leftRef;
			m_triangleRefStack.push_back(rightRef);
		}
	}

	o_leftSpec.m_numTris = leftEnd - leftStart;
	o_rightSpec.m_numTris = m_triangleRefStack.size() - rightStart;
}

void SBVH::splitReference(TriangleRef &o_leftSplitRef, TriangleRef &o_rightSplitRef, const TriangleRef &_triRef, const int &_axis, const float &_splitPosition)
{
	o_leftSplitRef.m_triIdx = o_rightSplitRef.m_triIdx = _triRef.m_triIdx;
	o_leftSplitRef.m_bb = o_rightSplitRef.m_bb = AABB();

	unsigned int *vertIdx = m_triangles[_triRef.m_triIdx].m_indices;
	for(size_t k = 0; k < 3; ++k)
	{
		// Get two consecutive verts
		ngl::Vec3 v1 = m_vertices[vertIdx[k]];
		ngl::Vec3 v2 = m_vertices[vertIdx[(k + 1)%3]];

		// We're only interested in the value of the selected axis of the vertices
		float v1pos = v1.m_openGL[_axis];
		float v2pos = v2.m_openGL[_axis];

		// Check whether the vertex is on the "left", "right" or at the split plane of the bin
		if(v1pos <= _splitPosition)
			o_leftSplitRef.m_bb.extendBB(v1);
		if(v1pos >= _splitPosition)
			o_rightSplitRef.m_bb.extendBB(v1);

		// Do we need to split the edge? E.g. does the edge go on both sides of the split plane
		if((v1pos < _splitPosition && v2pos > _splitPosition) ||
			 (v1pos > _splitPosition && v2pos < _splitPosition))
		{
			// Get the point of intersection with the split plane (linear interpolation)
			float alpha = (_splitPosition - v1pos) / (v2pos - v1pos);
			ngl::Vec3 intersectionPoint = vUtilities::lerp(v1, v2, alpha);

			// Extend both split references to the intersection point
			o_leftSplitRef.m_bb.extendBB(intersectionPoint);
			o_rightSplitRef.m_bb.extendBB(intersectionPoint);
		}
	}
}

void SBVH::sortTriRefStack(const unsigned int &_axis, const unsigned int &_first, const unsigned int &_last)
{
#ifdef BVH_DEBUG
	#ifdef BVH_DEBUG_SORT
		std::cout << "Triangle refs before sorting along " << (_axis == 0 ? 'X' : (_axis == 1 ? 'Y' : 'Z')) << "-axis:\n";
		for(size_t i = _first; i < _last; ++i)
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
		for(size_t i = _first; i < _last; ++i)
		{
			const TriangleRef &triRef = m_triangleRefStack[i];
			std::cout << "  Idx: " << triRef.m_triIdx << ", Centroid: " << (triRef.m_bb.minBounds().m_openGL[_axis] + triRef.m_bb.maxBounds().m_openGL[_axis])/2.f << "\n";
		}
		std::cout << "\n";
	#endif
#endif
}

void SBVH::insertionSort(const unsigned int &_axis, const unsigned int &_start, const unsigned int &_size)
{
	/// The following section is adapted from :-
	/// XXX [online].
	/// [Accessed 2017]. Available from: https://github.com/hhergeth/CudaTracerLib/blob/master/Engine/SceneBuilder/SplitBVHBuilder.cpp.
	for(size_t i = 1; i < _size; ++i)
	{
		int j = _start + i - 1;
		while(j >= static_cast<int>(_start) && compareBounds(_axis, j + 1, j) < 0)
		{
			vUtilities::swap(m_triangleRefStack[j], m_triangleRefStack[j + 1]);
			--j;
		}
	}
}

int SBVH::compareBounds(const unsigned int &_axis, const unsigned int &_a, const unsigned int &_b)
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
