#include <iostream>

#include "BVH.h"
#include "Utilities.h"

CacheFriendlyBVHNode* BVH::createBVH(const std::vector<vHVert> &_vertices, const std::vector<vHTriangle> &_triangles)
{
	std::vector<BoundingBox> workingTree;

	for(unsigned int i = 0; i < _triangles.size(); ++i)
	{
		const vHTriangle &tri = _triangles[i];
		BoundingBox bb;

		bb.m_triangles = &tri;

		// loop over triangle vertices and pick smallest vertex for bottom of triangle bbox
		bb.m_bottom = vUtilities::minvFloat3(bb.m_bottom, _vertices[_triangles[i].m_indices[0]].m_vert);
		bb.m_bottom = vUtilities::minvFloat3(bb.m_bottom, _vertices[_triangles[i].m_indices[1]].m_vert);
		bb.m_bottom = vUtilities::minvFloat3(bb.m_bottom, _vertices[_triangles[i].m_indices[2]].m_vert);

		// loop over triangle vertices and pick largest vertex for top of triangle bbox
		bb.m_top = vUtilities::maxvFloat3(bb.m_top, _vertices[_triangles[i].m_indices[0]].m_vert);
		bb.m_top = vUtilities::maxvFloat3(bb.m_top, _vertices[_triangles[i].m_indices[1]].m_vert);
		bb.m_top = vUtilities::maxvFloat3(bb.m_top, _vertices[_triangles[i].m_indices[2]].m_vert);

		// expand working list bbox by largest and smallest triangle bbox bounds
		m_bottom = vUtilities::minvFloat3(m_bottom, bb.m_bottom);
		m_top = vUtilities::maxvFloat3(m_top, bb.m_top);

		// compute triangle bbox center: (bbox top + bbox bottom) * 0.5
		bb.m_center = (bb.m_top + bb.m_bottom) * 0.5f;

		workingTree.push_back(bb);
	}

	BVHNode* root = recurseBoundingBoxes(workingTree);

	root->m_bottom = m_bottom;
	root->m_top = m_top;

	createCFBVH(root, _vertices, _triangles);

	return m_cfBVH;
}

BVHNode* BVH::recurseBoundingBoxes(const std::vector<BoundingBox> &_workingTree, unsigned int _depth)
{
	if(_workingTree.size() < 4)
	{
		BVHLeaf *leaf = new BVHLeaf;
		for(auto &bb : _workingTree)
		{
			leaf->m_triangles.push_back(bb.m_triangles);
			m_triCount++;
		}
		m_boxCount++;
		return leaf;
	}

	vFloat3 bottom = vFloat3(FLT_MAX, FLT_MAX, FLT_MAX);
	vFloat3 top = vFloat3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	// loop over all bboxes in current working list, expanding/growing the working list bbox
	for(auto &bb : _workingTree)
	{
		bottom = vUtilities::minvFloat3(bottom, bb.m_bottom);
		top = vUtilities::maxvFloat3(top, bb.m_top);
	}

	// SAH, surface area heuristic calculation
	// find surface area of bounding box by multiplying the dimensions of the working list's bounding box
	float side1 = top.x - bottom.x;  // length bbox along X-axis
	float side2 = top.y - bottom.y;  // length bbox along Y-axis
	float side3 = top.z - bottom.z;  // length bbox along Z-axis

	// the current bbox has a cost of (number of triangles) * surfaceArea of C = N * SA
	float minCost = _workingTree.size() * (side1*side2 + side2*side3 + side3*side1);

	// best split along axis, will indicate no split with better cost found (below)
	float bestSplit = FLT_MAX;
	int bestAxis = -1;

	// Try all 3 axises X, Y, Z
	// 0 = X, 1 = Y, 2 = Z axis
	for(unsigned int j = 0; j < 3; ++j)
	{
		int axis = j;

		// we will try dividing the triangles based on the current axis,
		// and we will try split values from "start" to "stop", one "step" at a time.
		float start;
		float stop;
		float step;

		// X-axis
		if(axis == 0)
		{
			start = bottom.x;
			stop = top.x;
		}
		// Y-axis
		else if(axis == 1)
		{
			start = bottom.y;
			stop = top.y;
		}
		// Z-axis
		else
		{
			start = bottom.z;
			stop = top.z;
		}

		// In that axis, do the bounding boxes in the work queue "span" across, (meaning distributed over a reasonable distance)?
		// Or are they all already "packed" on the axis? Meaning that they are too close to each other
		if(std::fabs(stop - start) < 1e-4)
			// BBox side along this axis too short, we must move to a different axis!
			continue; // go to next axis

		// Binning: Try splitting at a uniform sampling (at equidistantly spaced planes) that gets smaller the deeper we go:
		// size of "sampling grid": 1024 (depth 0), 512 (depth 1), etc
		// each bin has size "step"
		step = (stop - start) / (1024. / (_depth + 1.));

		// for each bin (equally spaced bins of size "step"):
		for(float testSplit = start + step; testSplit < stop - step; testSplit += step)
		{
			// Create left and right bounding box
			vFloat3 lbottom(FLT_MAX, FLT_MAX, FLT_MAX);
			vFloat3 ltop(-FLT_MAX, -FLT_MAX, -FLT_MAX);

			vFloat3 rbottom(FLT_MAX, FLT_MAX, FLT_MAX);
			vFloat3 rtop(-FLT_MAX, -FLT_MAX, -FLT_MAX);

			// The number of triangles in the left and right bboxes (needed to calculate SAH cost function)
			unsigned int countLeft = 0;
			unsigned int countRight = 0;

			// For each test split (or bin), allocate triangles in remaining work list based on their bbox centers
			// this is a fast O(N) pass, no triangle sorting needed (yet)
			for(auto &bb : _workingTree)
			{
				// compute bbox center
				float value;
				if(axis == 0)
					value = bb.m_center.x;		// X-axis
				else if(axis == 1)
					value = bb.m_center.y;		// Y-axis
				else
					value = bb.m_center.z;		// Z-axis

				if(value < testSplit)
				{
					// if center is smaller then testSplit value, put triangle in Left bbox
					lbottom = vUtilities::minvFloat3(lbottom, bb.m_bottom);
					ltop = vUtilities::maxvFloat3(ltop, bb.m_top);
					countLeft++;
				}
				else
				{
					// else put triangle in right bbox
					rbottom = vUtilities::minvFloat3(rbottom, bb.m_bottom);
					rtop = vUtilities::maxvFloat3(rtop, bb.m_top);
					countRight++;
				}
			}

			// Now use the Surface Area Heuristic to see if this split has a better "cost"

			// First, check for stupid partitionings, ie bins with 0 or 1 triangles make no sense
			if(countLeft <= 1 || countRight <= 1)
				continue;

			// It's a real partitioning, calculate the surface areas
			float lside1 = ltop.x - lbottom.x;
			float lside2 = ltop.y - lbottom.y;
			float lside3 = ltop.z - lbottom.z;

			float rside1 = rtop.x - rbottom.x;
			float rside2 = rtop.y - rbottom.y;
			float rside3 = rtop.z - rbottom.z;

			// calculate SurfaceArea of Left and Right BBox
			float surfaceLeft = lside1*lside2 + lside2*lside3 + lside3*lside1;
			float surfaceRight = rside1*rside2 + rside2*rside3 + rside3*rside1;

			// calculate total cost by multiplying left and right bbox by number of triangles in each
			float totalCost = surfaceLeft*countLeft + surfaceRight*countRight;

			// keep track of cheapest split found so far
			if(totalCost < minCost)
			{
				minCost = totalCost;
				bestSplit = testSplit;
				bestAxis = axis;
			}
		} // end of loop over all bins
	} // end of loop over all axises

	if(bestAxis == -1)
	{
		BVHLeaf *leaf = new BVHLeaf;
		for(auto &bb : _workingTree)
		{
			leaf->m_triangles.push_back(bb.m_triangles);
		}
		m_boxCount++;
		return leaf;
	}

	std::vector<BoundingBox> leftTree;
	std::vector<BoundingBox> rightTree;
	vFloat3 lbottom(FLT_MAX, FLT_MAX, FLT_MAX);
	vFloat3 ltop(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	vFloat3 rbottom(FLT_MAX, FLT_MAX, FLT_MAX);
	vFloat3 rtop(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	for(auto &bb : _workingTree)
	{
		// compute bbox center
		float value;
		if(bestAxis == 0)
			value = bb.m_center.x;
		else if(bestAxis == 1)
			value = bb.m_center.y;
		else
			value = bb.m_center.z;

		if(value < bestSplit)
		{ // add temporary bbox v from work list to left BBoxentries list,
			// becomes new working list of triangles in next step
			leftTree.push_back(bb);
			lbottom = vUtilities::minvFloat3(lbottom, bb.m_bottom);
			ltop = vUtilities::maxvFloat3(ltop, bb.m_top);
		}
		else {

			// Add triangle bbox v from working list to right BBoxentries,
			// becomes new working list of triangles in next step
			rightTree.push_back(bb);
			rbottom = vUtilities::minvFloat3(rbottom, bb.m_bottom);
			rtop = vUtilities::maxvFloat3(rtop, bb.m_top);
		}
	}

	BVHInner *inner = new BVHInner;
	inner->m_leftNode = recurseBoundingBoxes(leftTree, _depth + 1);
	inner->m_leftNode->m_bottom = lbottom;
	inner->m_leftNode->m_top = ltop;

	inner->m_rightNode = recurseBoundingBoxes(rightTree, _depth + 1);
	inner->m_rightNode->m_bottom = lbottom;
	inner->m_rightNode->m_top = ltop;

	m_boxCount++;

	return inner;
}

void BVH::createCFBVH(BVHNode* root, const std::vector<vHVert> &_vertices, const std::vector<vHTriangle> &_triangles)
{
	unsigned int cfTriCount = 0;
	unsigned int cfBoxCount = 0;

	m_triIndices = new unsigned int[m_triCount];
	m_cfBVH = new CacheFriendlyBVHNode[m_boxCount]; // array

	initCFBVH(root, &(_triangles[0]), cfBoxCount, cfTriCount);

	if((cfBoxCount != m_boxCount - 1) || (cfTriCount != m_triCount))
	{
		std::cerr << "Failed to create the cache friendly BVH.\n";
		exit(1);
	}

	std::cout << m_boxCount << "\n";
	for(unsigned int i = 0; i < m_boxCount; ++i)
	{
		if(m_cfBVH[i].m_u.m_leaf.m_count & 0x80000000)
		{
			std::cout << "Box " << i << ":\n";
			std::cout << "  BB Bot: " << m_cfBVH[i].m_bottom.x << ", " << m_cfBVH[i].m_bottom.y << ", " << m_cfBVH[i].m_bottom.z << "\n";
			std::cout << "  BB Top: " << m_cfBVH[i].m_top.x << ", " << m_cfBVH[i].m_top.y << ", " << m_cfBVH[i].m_top.z << "\n";
			std::cout << "  TriCount: " << (m_cfBVH[i].m_u.m_leaf.m_count ^ 0x80000000)  << "\n";
			std::cout << "  Start index: " << m_cfBVH[i].m_u.m_leaf.m_startIndexInTriIndexList << "\n\n";
		}
	}
}

void BVH::initCFBVH(BVHNode *root, const vHTriangle *_firstTri, unsigned &_cfBoxCount, unsigned &_cfTriCount)
{
	unsigned currIdxBoxes = _cfBoxCount;
	m_cfBVH[currIdxBoxes].m_bottom = root->m_bottom;
	m_cfBVH[currIdxBoxes].m_top = root->m_top;

	//DEPTH FIRST APPROACH (left first until complete)
	if(!root->isLeaf()) // inner node
	{
		BVHInner *p = dynamic_cast<BVHInner*>(root);
		// recursively populate left and right
		int idxLeft = ++_cfBoxCount;
		initCFBVH(p->m_leftNode, _firstTri, _cfBoxCount, _cfTriCount);

		int idxRight = ++_cfBoxCount;
		initCFBVH(p->m_rightNode, _firstTri, _cfBoxCount, _cfTriCount);

		m_cfBVH[currIdxBoxes].m_u.m_inner.m_leftIndex = idxLeft;
		m_cfBVH[currIdxBoxes].m_u.m_inner.m_rightIndex = idxRight;
	}
	else // leaf
	{
		BVHLeaf *p = dynamic_cast<BVHLeaf*>(root);
		unsigned int count = (unsigned int)p->m_triangles.size();
		m_cfBVH[currIdxBoxes].m_u.m_leaf.m_count = 0x80000000 | count;  // highest bit set indicates a leaf node (inner node if highest bit is 0)
		m_cfBVH[currIdxBoxes].m_u.m_leaf.m_startIndexInTriIndexList = _cfTriCount;

		for(auto &tri : p->m_triangles)
		{
			m_triIndices[_cfTriCount++] = tri - _firstTri;
		}
	}
}
