#include <assert.h>
#include <iostream>

#include "SBVH.h"

#define SBVH_DEBUG_ON
//#define SBVH_DEBUG_OBJECT_SPLIT
//#define SBVH_DEBUG_AXIS_SORT

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

	m_minOverlap = rootSpecification.m_bounds.surfaceArea() * 5e-5;
	m_root = buildSBVH(rootSpecification, 0);

#ifdef SBVH_DEBUG_ON
	std::cout << "\nSBVH STATS\n";
	std::cout << "---------------\n";
	std::cout << "Root bounds:\n";
	std::cout << "  Min: " << rootSpecification.m_bounds.minBounds().m_x << ", " << rootSpecification.m_bounds.minBounds().m_y << ", " << rootSpecification.m_bounds.minBounds().m_z << "\n";
	std::cout << "  Max: " << rootSpecification.m_bounds.maxBounds().m_x << ", " << rootSpecification.m_bounds.maxBounds().m_y << ", " << rootSpecification.m_bounds.maxBounds().m_z << "\n";
	std::cout << "  Min overlap: " << m_minOverlap << "\n\n";

	std::cout << "SBVH Tree: \n";
	std::cout << "  Nodes: " << m_root->nodeCount() << "\n";

	std::cout << "---------------\n\n";
#endif
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
	ObjectSplitCandidate objectCandidate = findObjectSplitCandidate(_nSpec, nodeCost);
	SpatialSplitCandidate spatialCandidate;

	if(_depth < kMaxDepth)
	{
		AABB overlap = objectCandidate.m_leftBound;
		overlap.intersectBB(objectCandidate.m_rightBound);

		if(overlap.surfaceArea() >= m_minOverlap)
		{
			spatialCandidate = findSpatialSplitCandidate(_nSpec, nodeCost);
		}
	}

	float minCost = std::min(leafCost, std::min(objectCandidate.m_cost, spatialCandidate.m_cost));
	if(minCost == leafCost/* && _nSpec.m_numRef <= kMaxLeafSize*/)
	{
		return createLeafNode(_nSpec);
	}

	NodeSpec leftSpec;
	NodeSpec rightSpec;
	if(minCost == spatialCandidate.m_cost)
	{
		performSpatialSplit(leftSpec, rightSpec, _nSpec, spatialCandidate);
	}
	if(!leftSpec.m_numRef || !rightSpec.m_numRef)
	{
		performObjectSplit(leftSpec, rightSpec, _nSpec, objectCandidate);
	}

	SBVHNode *leftChild = buildSBVH(leftSpec, _depth + 1);
	SBVHNode *rightChild = buildSBVH(rightSpec, _depth + 1);

	return new InnerNode(_nSpec.m_bounds, leftChild, rightChild);
}

SBVHNode* SBVH::createLeafNode(const NodeSpec &_nSpec)
{
	for(unsigned int i = 0; i < _nSpec.m_numRef; ++i)
	{
		m_triIndices.push_back(m_triRefStack.back().m_triIndex);
		m_triRefStack.pop_back();
	}

	return new LeafNode(_nSpec.m_bounds, m_triIndices.size() - _nSpec.m_numRef, m_triIndices.size());
}

SBVH::ObjectSplitCandidate SBVH::findObjectSplitCandidate(const NodeSpec &_nSpec, const float &_nodeCost)
{
	ObjectSplitCandidate candidate;

	unsigned int firstRefIndex = m_triRefStack.size() - _nSpec.m_numRef;
	AABB *leftBounds = new AABB[_nSpec.m_numRef];

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
		AABB lBound;
		for(unsigned int i = 0; i < _nSpec.m_numRef; ++i)
		{
			lBound.extendBB(m_triRefStack[firstRefIndex + i].m_bounds);
			leftBounds[i] = lBound;
//			m_rightBounds[i - 1] = rightBounds;
		}

		/*
		 * Sweep from right
		 */
		AABB rBound;
		for(unsigned int i = _nSpec.m_numRef - 1; i > 0; --i)
		{
			rBound.extendBB(m_triRefStack[firstRefIndex + i].m_bounds);
			float cost = _nodeCost + rBound.surfaceArea() * kTriangleCost * i + leftBounds[i].surfaceArea() * kTriangleCost * (_nSpec.m_numRef - i);
			if(cost < candidate.m_cost)
			{
				candidate.m_cost = cost;
				candidate.m_axis = axis;
				candidate.m_leftTris = i;
				candidate.m_leftBound = leftBounds[i];
				candidate.m_rightBound = rBound;
			}
		}
	}

	delete [] leftBounds;

#ifdef SBVH_DEBUG_ON
	#ifdef SBVH_DEBUG_OBJECT_SPLIT
		std::cout << "\nObject split candidate:\n";
		std::cout << "\tNum tris: " << _nSpec.m_numRef << "\n";
		std::cout << "\tBest cost: " << candidate.m_cost << "\n";
		std::cout << "\tBest axis: " << (candidate.m_axis == -1 ? "Not found" : (candidate.m_axis == 0 ? "X" : (candidate.m_axis == 1 ? "Y" : "Z"))) << "\n";
	#endif
#endif

	return candidate;
}

SBVH::SpatialSplitCandidate SBVH::findSpatialSplitCandidate(const NodeSpec &_nSpec, const float &_nodeCost)
{
	// Initialize bins.
	ngl::Vec3 origin = _nSpec.m_bounds.minBounds();
	ngl::Vec3 binSize = (_nSpec.m_bounds.maxBounds() - origin) * (1.0f / kSpatialBins);
	ngl::Vec3 invBinSize = ngl::Vec3(1.0f / binSize.m_x, 1.0f / binSize.m_y, 1.0f / binSize.m_z);

	for(unsigned int axis = 0; axis < 3; ++axis)
	{
		for(unsigned int i = 0; i < kSpatialBins; ++i)
		{
			SpatialBin &bin = m_bins[axis][i];
			bin.m_bounds = AABB();
			bin.m_enter = 0;
			bin.m_exit = 0;
		}
	}

	// Chop references into bins.
	for(unsigned int refIdx = m_triRefStack.size() - _nSpec.m_numRef; refIdx < m_triRefStack.size(); ++refIdx)
	{
		const TriRef &ref = m_triRefStack[refIdx];

		ngl::Vec3 firstBin = ngl::Vec3((ref.m_bounds.minBounds() - origin) * invBinSize);
		vUtilities::clampVec3(firstBin, ngl::Vec3(0, 0, 0), ngl::Vec3(kSpatialBins - 1, kSpatialBins - 1, kSpatialBins - 1));
		ngl::Vec3 lastBin = ngl::Vec3((ref.m_bounds.maxBounds() - origin) * invBinSize);
		vUtilities::clampVec3(lastBin, firstBin, ngl::Vec3(kSpatialBins - 1, kSpatialBins - 1, kSpatialBins - 1));

		for(unsigned int axis = 0; axis < 3; ++axis)
		{
			TriRef currRef = ref;
			for(unsigned int i = firstBin.m_openGL[axis]; i < lastBin.m_openGL[axis]; ++i)
			{
				TriRef leftRef;
				TriRef rightRef;
				splitReference(leftRef, rightRef, currRef, axis, origin.m_openGL[axis] + binSize.m_openGL[axis] * static_cast<float>(i + 1));

				m_bins[axis][i].m_bounds.extendBB(leftRef.m_bounds);

				currRef = rightRef;
			}
			m_bins[axis][static_cast<int>(lastBin.m_openGL[axis])].m_bounds.extendBB(currRef.m_bounds);
			m_bins[axis][static_cast<int>(firstBin.m_openGL[axis])].m_enter++;
			m_bins[axis][static_cast<int>(lastBin.m_openGL[axis])].m_exit++;
		}
	}

	// Select best split plane.
	SpatialSplitCandidate candidate;
	AABB *rightBounds = new AABB[kSpatialBins - 1];

	// Loop through all the axes, X = 0, Y = 1, Z = 2
	for(unsigned int axis = 0; axis < 3; ++axis)
	{
		AABB rBound;
		for(unsigned int i = kSpatialBins - 1; i > 0; --i)
		{
			rBound.extendBB(m_bins[axis][i].m_bounds);
			rightBounds[i - 1] = rBound;
		}

		// Sweep left to right and select lowest SAH.

		AABB lBound;
		int leftTris = 0;
		int rightTris = _nSpec.m_numRef;

		for(unsigned int i = 1; i < kSpatialBins; ++i)
		{
			lBound.extendBB(m_bins[axis][i - 1].m_bounds);
			leftTris += m_bins[axis][i - 1].m_enter;
			rightTris -= m_bins[axis][i - 1].m_exit;

			float cost = _nodeCost + lBound.surfaceArea() * kTriangleCost * leftTris + rightBounds[i - 1].surfaceArea() * kTriangleCost * rightTris;
			if(cost < candidate.m_cost)
			{
				candidate.m_cost = cost;
				candidate.m_axis = axis;
				candidate.m_location = origin.m_openGL[axis] + binSize.m_openGL[axis] * static_cast<float>(i);
			}
		}
	}
	delete [] rightBounds;

	return candidate;
}

void SBVH::splitReference(TriRef &o_leftRef, TriRef &o_rightRef, const TriRef &_ref, const unsigned int &_axis, const float &_pos)
{
	// Initialize references.
	o_leftRef.m_triIndex = o_rightRef.m_triIndex = _ref.m_triIndex;
	o_leftRef.m_bounds = o_rightRef.m_bounds = AABB();

	// Loop over vertices/edges.
	const unsigned *inds = m_triangles[_ref.m_triIndex].m_indices;
	ngl::Vec3 &v1 = m_vertices[inds[2]];

	for(unsigned int i = 0; i < 3; ++i)
	{
		const ngl::Vec3 &v0 = v1;
		v1 = m_vertices[inds[i]];
		float v0p = v0.m_openGL[_axis];
		float v1p = v1.m_openGL[_axis];

		// Insert vertex to the boxes it belongs to.
		if(v0p <= _pos)
			o_leftRef.m_bounds.extendBB(v0);
		if(v0p >= _pos)
			o_rightRef.m_bounds.extendBB(v0);

		// Edge intersects the plane => insert intersection to both boxes.
		if((v0p < _pos && v1p > _pos) || (v0p > _pos && v1p < _pos))
		{
			ngl::Vec3 t = vUtilities::lerp(v0, v1, vUtilities::clamp((_pos - v0p) / (v1p - v0p), 0.0f, 1.0f));
			o_leftRef.m_bounds.extendBB(t);
			o_rightRef.m_bounds.extendBB(t);
		}
	}

	// Intersect with original bounds.
	o_leftRef.m_bounds.setMaxBoundsComponent(_axis, _pos);
	o_rightRef.m_bounds.setMinBoundsComponent(_axis, _pos);
	o_leftRef.m_bounds.intersectBB(_ref.m_bounds);
	o_rightRef.m_bounds.intersectBB(_ref.m_bounds);
}

void SBVH::performSpatialSplit(NodeSpec &io_leftSpec, NodeSpec &io_rightSpec, const NodeSpec &_nSpec, const SpatialSplitCandidate &_split)
{
	// Categorize references and compute bounds.
	//
	// Left-hand side:      [leftStart, leftEnd[
	// Uncategorized/split: [leftEnd, rightStart[
	// Right-hand side:     [rightStart, m_triRefStack.size()[

	unsigned int leftStart = m_triRefStack.size() - _nSpec.m_numRef;
	unsigned int leftEnd = leftStart;
	unsigned int rightStart = m_triRefStack.size();
	io_leftSpec.m_bounds = io_rightSpec.m_bounds = AABB();

	for(unsigned int i = leftEnd; i < rightStart; ++i)
	{
		// Entirely on the left-hand side?
		if(m_triRefStack[i].m_bounds.maxBounds().m_openGL[_split.m_axis] <= _split.m_location)
		{
			io_leftSpec.m_bounds.extendBB(m_triRefStack[i].m_bounds);
			vUtilities::swap(m_triRefStack[i], m_triRefStack[leftEnd++]);
		}
		// Entirely on the right-hand side?
		else if(m_triRefStack[i].m_bounds.minBounds().m_openGL[_split.m_axis] >= _split.m_location)
		{
			io_rightSpec.m_bounds.extendBB(m_triRefStack[i].m_bounds);
			vUtilities::swap(m_triRefStack[i--], m_triRefStack[--rightStart]);
		}
	}

	// Duplicate or unsplit references intersecting both sides.
	while(leftEnd < rightStart)
	{
		// Split reference.
		TriRef lref;
		TriRef rref;
		splitReference(lref, rref, m_triRefStack[leftEnd], _split.m_axis, _split.m_location);

		// Compute SAH for duplicate/unsplit candidates.
		AABB lub = io_leftSpec.m_bounds;  // Unsplit to left:     new left-hand bounds.
		AABB rub = io_rightSpec.m_bounds; // Unsplit to right:    new right-hand bounds.
		AABB ldb = io_leftSpec.m_bounds;  // Duplicate:           new left-hand bounds.
		AABB rdb = io_rightSpec.m_bounds; // Duplicate:           new right-hand bounds.
		lub.extendBB(m_triRefStack[leftEnd].m_bounds);
		rub.extendBB(m_triRefStack[leftEnd].m_bounds);
		ldb.extendBB(lref.m_bounds);
		rdb.extendBB(rref.m_bounds);

		float lac = kTriangleCost * (leftEnd - leftStart);
		float rac = kTriangleCost * (m_triRefStack.size() - rightStart);
		float lbc = kTriangleCost * (leftEnd - leftStart + 1);
		float rbc = kTriangleCost * (m_triRefStack.size() - rightStart + 1);

		float unsplitLeftSAH = lub.surfaceArea() * lbc + io_rightSpec.m_bounds.surfaceArea() * rac;
		float unsplitRightSAH = io_leftSpec.m_bounds.surfaceArea() * lac + rub.surfaceArea() * rbc;
		float duplicateSAH = ldb.surfaceArea() * lbc + rdb.surfaceArea() * rbc;
		float minSAH = std::min(unsplitLeftSAH, std::min(unsplitRightSAH, duplicateSAH));

		// Unsplit to left?
		if (minSAH == unsplitLeftSAH)
		{
			io_leftSpec.m_bounds = lub;
			leftEnd++;
		}
		// Unsplit to right?
		else if(minSAH == unsplitRightSAH)
		{
			io_rightSpec.m_bounds = rub;
			vUtilities::swap(m_triRefStack[leftEnd], m_triRefStack[--rightStart]);
		}
		// Duplicate?
		else
		{
			io_leftSpec.m_bounds = ldb;
			io_rightSpec.m_bounds = rdb;
			m_triRefStack[leftEnd++] = lref;
			m_triRefStack.push_back(rref);
		}
	}

	io_leftSpec.m_numRef = leftEnd - leftStart;
	io_rightSpec.m_numRef = m_triRefStack.size() - rightStart;
}

void SBVH::performObjectSplit(NodeSpec &o_leftSpec, NodeSpec &o_rightSpec, const NodeSpec &_nSpec, const ObjectSplitCandidate &_object)
{
	/*
	 * sort S in axis ‘bestAxis’
	 * S1 = S {0..bestEvent); S2 = S[bestEvent..|S|)
	 */
	sortTriRefStack(_object.m_axis, m_triRefStack.size() - _nSpec.m_numRef, m_triRefStack.size(), true);

	o_leftSpec.m_numRef = _object.m_leftTris;
	o_leftSpec.m_bounds = _object.m_leftBound;
	o_rightSpec.m_numRef = _nSpec.m_numRef - _object.m_leftTris;
	o_rightSpec.m_bounds = _object.m_rightBound;
}

void SBVH::sortTriRefStack(const unsigned int &_axis, const unsigned int &_first, const unsigned int &_last, bool orig)
{
#ifdef SBVH_DEBUG_ON
	#ifdef SBVH_DEBUG_AXIS_SORT
	if(_first == 0 && _last == m_triRefStack.size() && orig)
	{
		std::cout << "\nTriangle indices before sorting by axis " << _axis << " centroids:\n  ";
		for(unsigned int k = 0; k < m_triRefStack.size(); ++k)
		{
			const TriRef &triA = m_triRefStack[k];
			std::cout << "[i: " << triA.m_triIndex << ", c: " << triA.m_bounds.minBounds().m_openGL[_axis] + triA.m_bounds.maxBounds().m_openGL[_axis] << "] ";
		}
		std::cout << "\n";
	}
	#endif
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
	#ifdef SBVH_DEBUG_AXIS_SORT
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
