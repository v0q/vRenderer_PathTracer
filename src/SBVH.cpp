#include <assert.h>
#include <iostream>

#include "SBVH.h"
#include "Utilities.h"

SBVH::SBVH(std::vector<vHTriangle> &_tris, std::vector<vFloat3> &_verts, const float &_sA) :
	m_triangles(_tris),
	m_vertices(_verts),
	m_numDuplicates(0),
	m_sortDim(0),
	m_minOverlap(0.f),
	m_splitAlpha(_sA),
	m_initialised(true)
{
	exec();
}

SBVHNode* SBVH::exec()
{
	assert(m_initialised);

	// See SBVH paper by Martin Stich for details
	// Initialize reference stack and determine root bounds.

	NodeSpec rootSpec;
	rootSpec.m_numRef = m_triangles.size();  // number of triangles/references in entire scene (root)
	m_refStack.resize(rootSpec.m_numRef);

	// calculate the bounds of the rootnode by merging the AABBs of all the references
	for(unsigned int i = 0; i < rootSpec.m_numRef; ++i)
	{
		// assign triangle to the array of references
		m_refStack[i].m_triIdx = i;

		// grow the bounds of each reference AABB in all 3 dimensions by including the vertex
		for(unsigned int j = 0; j < 3; ++j)
		{
			m_refStack[i].m_bounds.grow(m_vertices[m_triangles[i].m_indices[j]]);
		}

		rootSpec.m_bounds.grow(m_refStack[i].m_bounds);
	}

	// Initialize rest of the members.

	// split alpha (maximum allowable overlap) relative to size of rootnode
	m_minOverlap = rootSpec.m_bounds.area() * m_splitAlpha;
	m_rightBounds.resize(std::max(rootSpec.m_numRef, static_cast<unsigned int>(NumSpatialBins)) - 1);

	// Build recursively.
	SBVHNode *root = buildNode(rootSpec, 0);

//	m_bvh.getTriIndices().compact();   // removes unused memoryspace from triIndices array

//	if (m_params.enablePrints)
//		printf("SplitBVHBuilder: progress %.0f%%, duplicates %.0f%%\n",
//		100.0f, (float)m_numDuplicates / (float)m_bvh.getScene()->getNumTriangles() * 100.0f);

	std::cout << "Root spec references: " << rootSpec.m_numRef << "\n";
	std::cout << "Root spec bounds:\n\tMin: " << rootSpec.m_bounds.minBounds().x << ", " << rootSpec.m_bounds.minBounds().y << ", " << rootSpec.m_bounds.minBounds().z << "\n";
	std::cout << "\tMax: " << rootSpec.m_bounds.maxBounds().x << ", " << rootSpec.m_bounds.maxBounds().y << ", " << rootSpec.m_bounds.maxBounds().z << "\n";
	std::cout << "Triangle indices: " << m_triIndices.size() << "\n";

	return root;
}

int SBVH::sortCompare(void *io_data, const unsigned int &_idxA, const unsigned int &_idxB)
{
	const SBVH *ptr = (const SBVH*)io_data;

	int dim = ptr->m_sortDim;

	if(_idxA >= ptr->m_refStack.size() || _idxB >= ptr->m_refStack.size())
		return -1;

	const Reference& ra = ptr->m_refStack[_idxA];
	const Reference& rb = ptr->m_refStack[_idxB];

	float ca = ra.m_bounds.minBounds().v[dim] + ra.m_bounds.maxBounds().v[dim];
	float cb = rb.m_bounds.minBounds().v[dim] + rb.m_bounds.maxBounds().v[dim];

	return (ca < cb) ? -1 : (ca > cb) ? 1 : (ra.m_triIdx < rb.m_triIdx) ? -1 : (ra.m_triIdx > rb.m_triIdx) ? 1 : 0;
}

void SBVH::sortSwap(void *io_data, const unsigned int &_idxA, const unsigned int &_idxB)
{
	SBVH *ptr = (SBVH*)io_data;
	vUtilities::swap(ptr->m_refStack[_idxA], ptr->m_refStack[_idxB]);
}

SBVHNode* SBVH::buildNode(const NodeSpec &_spec, const unsigned int &_level)
{
	if(_spec.m_numRef <= m_platform.getMinLeafSize() || _level >= MaxDepth)
	{
		std::cout << "Here\n";
		return createLeaf(_spec);
	}
	std::cout << "Got here 1 in level" << _level << "\n";

	// Find split candidates.
	float area = _spec.m_bounds.area();
	float leafSAH = area * m_platform.getTriangleCost(_spec.m_numRef);
	float nodeSAH = area * m_platform.getNodeCost(2);
	std::cout << "Got here 2 in level" << _level << " " << leafSAH << " " << nodeSAH << "\n";
	ObjectSplit object = findObjectSplit(_spec, nodeSAH);

	SpatialSplit spatial;
	if(_level < MaxSpatialDepth)
	{
		AABB overlap = object.m_leftBounds;
		overlap.intersect(object.m_rightBounds);
		if(overlap.area() >= m_minOverlap)
			spatial = findSpatialSplit(_spec, nodeSAH);
	}

	// Leaf SAH is the lowest => create leaf.
	float minSAH = vUtilities::min3f(leafSAH, object.m_sah, spatial.m_sah);

	if(minSAH == leafSAH && _spec.m_numRef <= m_platform.getMaxLeafSize())
	{
		return createLeaf(_spec);
	}
	std::cout << "Got here 4 in level" << _level << "\n";

	// Leaf SAH is not the lowest => Perform spatial split.
	NodeSpec left;
	NodeSpec right;
	if(minSAH == spatial.m_sah)
	{
		performSpatialSplit(left, right, _spec, spatial);
	}

	std::cout << "Got here 5 in level" << _level << "\n";

	// if either child contains no triangles/references
	if(!left.m_numRef || !right.m_numRef)
	{
		performObjectSplit(left, right, _spec, object);
	}

	std::cout << "Got here 6 in level" << _level << "\n";

	// Create inner node.
	m_numDuplicates += left.m_numRef + right.m_numRef - _spec.m_numRef;

	SBVHNode* rightNode = buildNode(right, _level + 1);
	SBVHNode* leftNode = buildNode(left, _level + 1);

	return new InnerNode(_spec.m_bounds, leftNode, rightNode);
}

SBVHNode* SBVH::createLeaf(const NodeSpec &_spec)
{
	std::cout << "Creating a leaf " << _spec.m_numRef << "\n";
	// take a triangle from the stack and add it to tris array
	for(unsigned int i = 0; i < _spec.m_numRef; ++i)
	{
		m_triIndices.push_back(m_refStack.back().m_triIdx);
		m_refStack.pop_back();
	}

	return new LeafNode(_spec.m_bounds, m_triIndices.size() - _spec.m_numRef, m_triIndices.size());
}

SBVH::ObjectSplit SBVH::findObjectSplit(const NodeSpec &_spec, const float &_nodeSAH)
{
	ObjectSplit split;
	const Reference* refPtr = &(m_refStack[m_refStack.size() - _spec.m_numRef]);
	std::cout << "Got here!\n";

	// Sort along each dimension.
	for(m_sortDim = 0; m_sortDim < 3; ++m_sortDim)
	{
		vUtilities::Sort(m_refStack.size() - _spec.m_numRef, m_refStack.size(), this, sortCompare, sortSwap);

		// Sweep right to left and determine bounds.

		AABB rightBounds;
		for(unsigned int j = _spec.m_numRef - 1; j > 0; --j)
		{
			rightBounds.grow(refPtr[j].m_bounds);
			m_rightBounds[j - 1] = rightBounds;
		}

		// Sweep left to right and select lowest SAH.

		AABB leftBounds;
		for(unsigned int j = 1; j < _spec.m_numRef; ++j)
		{
			leftBounds.grow(refPtr[j - 1].m_bounds);
			float sah = _nodeSAH + leftBounds.area() * m_platform.getTriangleCost(j) + m_rightBounds[j - 1].area() * m_platform.getTriangleCost(_spec.m_numRef - j);
			if(sah < split.m_sah)
			{
				split.m_sah = sah;
				split.m_sortDim = m_sortDim;
				split.m_numLeft = j;
				split.m_leftBounds = leftBounds;
				split.m_rightBounds = m_rightBounds[j - 1];
			}
		}
	}

	return split;
}

void SBVH::performObjectSplit(NodeSpec &_left, NodeSpec &_right, const NodeSpec &_spec, const ObjectSplit &_split)
{
	m_sortDim = _split.m_sortDim;
	vUtilities::Sort(m_refStack.size() - _spec.m_numRef, m_refStack.size(), this, sortCompare, sortSwap);

	_left.m_numRef = _split.m_numLeft;
	_left.m_bounds = _split.m_leftBounds;
	_right.m_numRef = _spec.m_numRef - _split.m_numLeft;
	_right.m_bounds = _split.m_rightBounds;
}

SBVH::SpatialSplit SBVH::findSpatialSplit(const NodeSpec &_spec, const float &_nodeSAH)
{
	// Initialize bins.
	vFloat3 origin = _spec.m_bounds.minBounds();
	vFloat3 binSize = (_spec.m_bounds.maxBounds() - origin) * (1.0f / (float)NumSpatialBins);
	vFloat3 invBinSize = vFloat3(1.0f / binSize.x, 1.0f / binSize.y, 1.0f / binSize.z);

	for(unsigned int dim = 0; dim < 3; ++dim)
	{
		for(unsigned int i = 0; i < NumSpatialBins; i++)
		{
			SpatialBin& bin = m_bins[dim][i];
			bin.m_bounds = AABB();
			bin.m_enter = 0;
			bin.m_exit = 0;
		}
	}

	// Chop references into bins.

	for(unsigned int refIdx = m_refStack.size() - _spec.m_numRef; refIdx < m_refStack.size(); refIdx++)
	{
		const Reference& ref = m_refStack[refIdx];

		vInt3 firstBin = vUtilities::clamp(vInt3((ref.m_bounds.minBounds() - origin) * invBinSize), vInt3(0, 0, 0), vInt3(NumSpatialBins - 1, NumSpatialBins - 1, NumSpatialBins - 1));
		vInt3 lastBin = vUtilities::clamp(vInt3((ref.m_bounds.maxBounds() - origin) * invBinSize), firstBin, vInt3(NumSpatialBins - 1, NumSpatialBins - 1, NumSpatialBins - 1));

		for(int dim = 0; dim < 3; ++dim)
		{
			Reference currRef = ref;
			for(int i = firstBin.v[dim]; i < lastBin.v[dim]; ++i)
			{
				Reference leftRef;
				Reference rightRef;
				splitReference(leftRef, rightRef, currRef, dim, origin.v[dim] + binSize.v[dim] * (float)(i + 1));
				m_bins[dim][i].m_bounds.grow(leftRef.m_bounds);
				currRef = rightRef;
			}
			m_bins[dim][lastBin.v[dim]].m_bounds.grow(currRef.m_bounds);
			m_bins[dim][firstBin.v[dim]].m_enter++;
			m_bins[dim][lastBin.v[dim]].m_exit++;
		}
	}

	// Select best split plane.

	SpatialSplit split;
	for(int dim = 0; dim < 3; ++dim)
	{
		// Sweep right to left and determine bounds.
		AABB rightBounds;
		for(int i = NumSpatialBins - 1; i > 0; --i)
		{
			rightBounds.grow(m_bins[dim][i].m_bounds);
			m_rightBounds[i - 1] = rightBounds;
		}

		// Sweep left to right and select lowest SAH.

		AABB leftBounds;
		int leftNum = 0;
		int rightNum = _spec.m_numRef;

		for(unsigned int i = 1; i < NumSpatialBins; ++i)
		{
			leftBounds.grow(m_bins[dim][i - 1].m_bounds);
			leftNum += m_bins[dim][i - 1].m_enter;
			rightNum -= m_bins[dim][i - 1].m_exit;

			float sah = _nodeSAH + leftBounds.area() * m_platform.getTriangleCost(leftNum) + m_rightBounds[i - 1].area() * m_platform.getTriangleCost(rightNum);
			if(sah < split.m_sah)
			{
				split.m_sah = sah;
				split.m_dim = dim;
				split.m_pos = origin.v[dim] + binSize.v[dim] * (float)i;
			}
		}
	}
	return split;
}

void SBVH::performSpatialSplit(NodeSpec &_left, NodeSpec &_right, const NodeSpec &_spec, const SpatialSplit &_split)
{
	// Categorize references and compute bounds.
	//
	// Left-hand side:      [leftStart, leftEnd[
	// Uncategorized/split: [leftEnd, rightStart[
	// Right-hand side:     [rightStart, refs.size()[

	unsigned int leftStart = m_refStack.size() - _spec.m_numRef;
	unsigned int leftEnd = leftStart;
	unsigned int rightStart = m_refStack.size();
	_left.m_bounds = _right.m_bounds = AABB();

	for(unsigned int i = leftEnd; i < rightStart; ++i)
	{
		// Entirely on the left-hand side?
		if(m_refStack[i].m_bounds.maxBounds().v[_split.m_dim] <= _split.m_pos)
		{
			_left.m_bounds.grow(m_refStack[i].m_bounds);
			vUtilities::swap(m_refStack[i], m_refStack[leftEnd++]);
		}
		// Entirely on the right-hand side?
		else if(m_refStack[i].m_bounds.minBounds().v[_split.m_dim] >= _split.m_pos)
		{
			_right.m_bounds.grow(m_refStack[i].m_bounds);
			vUtilities::swap(m_refStack[i--], m_refStack[--rightStart]);
		}
	}

	// Duplicate or unsplit references intersecting both sides.
	while(leftEnd < rightStart)
	{
		// Split reference.

		Reference lref;
		Reference rref;
		splitReference(lref, rref, m_refStack[leftEnd], _split.m_dim, _split.m_pos);

		// Compute SAH for duplicate/unsplit candidates.

		AABB lub = _left.m_bounds;  // Unsplit to left:     new left-hand bounds.
		AABB rub = _right.m_bounds; // Unsplit to right:    new right-hand bounds.
		AABB ldb = _left.m_bounds;  // Duplicate:           new left-hand bounds.
		AABB rdb = _right.m_bounds; // Duplicate:           new right-hand bounds.
		lub.grow(m_refStack[leftEnd].m_bounds);
		rub.grow(m_refStack[leftEnd].m_bounds);
		ldb.grow(lref.m_bounds);
		rdb.grow(rref.m_bounds);

		float lac = m_platform.getTriangleCost(leftEnd - leftStart);
		float rac = m_platform.getTriangleCost(m_refStack.size() - rightStart);
		float lbc = m_platform.getTriangleCost(leftEnd - leftStart + 1);
		float rbc = m_platform.getTriangleCost(m_refStack.size() - rightStart + 1);

		float unsplitLeftSAH = lub.area() * lbc + _right.m_bounds.area() * rac;
		float unsplitRightSAH = _left.m_bounds.area() * lac + rub.area() * rbc;
		float duplicateSAH = ldb.area() * lbc + rdb.area() * rbc;
		float minSAH = vUtilities::min3f(unsplitLeftSAH, unsplitRightSAH, duplicateSAH);

		// Unsplit to left?
		if(minSAH == unsplitLeftSAH)
		{
			_left.m_bounds = lub;
			leftEnd++;
		}
		// Unsplit to right?
		else if(minSAH == unsplitRightSAH)
		{
			_right.m_bounds = rub;
			vUtilities::swap(m_refStack[leftEnd], m_refStack[--rightStart]);
		}
		// Duplicate?
		else
		{
			_left.m_bounds = ldb;
			_right.m_bounds = rdb;
			m_refStack[leftEnd++] = lref;
			m_refStack.push_back(rref);
		}
	}

	_left.m_numRef = leftEnd - leftStart;
	_right.m_numRef = m_refStack.size() - rightStart;
}

void SBVH::splitReference(Reference &_left, Reference &_right, const Reference &_ref, const unsigned int &_dim, const float &_pos)
{
	// Initialize references.
	_left.m_triIdx = _right.m_triIdx = _ref.m_triIdx;
	_left.m_bounds = _right.m_bounds = AABB();

	// Loop over vertices/edges.
	const vInt3 inds = m_triangles[_ref.m_triIdx].m_indices;
	const vFloat3 *verts = &m_vertices[0];
	const vFloat3 *v1 = &verts[inds.z];

	for(unsigned int i = 0; i < 3; ++i)
	{
		const vFloat3 *v0 = v1;
		v1 = &verts[inds.v[i]];
		float v0p = (*v0).v[_dim];
		float v1p = (*v1).v[_dim];

		// Insert vertex to the boxes it belongs to.

		if(v0p <= _pos)
			_left.m_bounds.grow(*v0);
		if(v0p >= _pos)
			_right.m_bounds.grow(*v0);

		// Edge intersects the plane => insert intersection to both boxes.
		if((v0p < _pos && v1p > _pos) || (v0p > _pos && v1p < _pos))
		{
			vFloat3 t = vUtilities::lerp(*v0, *v1, vUtilities::clamp((_pos - v0p) / (v1p - v0p), 0.0f, 1.0f));
			_left.m_bounds.grow(t);
			_right.m_bounds.grow(t);
		}
	}

	// Intersect with original bounds.
	_left.m_bounds.setMaxBoundsComponent(_dim, _pos);
	_right.m_bounds.setMinBoundsComponent(_dim, _pos);
	_left.m_bounds.intersect(_ref.m_bounds);
	_right.m_bounds.intersect(_ref.m_bounds);
}
