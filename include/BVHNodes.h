///
/// \file BVHNodes.h
/// \brief Simple node class(es) for the SBVH tree
/// \authors Teemu Lindborg
/// \version 1.0
/// \date 04/05/17 Updated to NCCA Coding standard
/// Revision History :
/// Initial Version 08/12/16
/// \todo -
///

#pragma once

#include <assert.h>
#include "AABB.h"

///
/// \brief The BVHNode class Abstract base class for the nodes
///
class BVHNode
{
public:
	///
	/// \brief BVHNode Empty default ctor,
	///
	BVHNode() {}

	///
	/// \brief ~BVHNode Default virtual dtor
	///
	virtual ~BVHNode() {}

	///
	/// \brief childNode Virtual method to fetch child nodes of the current node
	/// \param _index Index of the child node to get
	/// \return Child node at given index
	///
	virtual BVHNode* childNode(const unsigned int &_index) const = 0;

	///
	/// \brief isLeaf Whether the current node is a leaf node or not
	/// \return True if a leaf node
	///
	virtual bool isLeaf() const = 0;

	///
	/// \brief numChildNodes Get the number of child nodes for the current node
	/// \return Number of immediate child nodes
	///
	virtual unsigned int numChildNodes() const = 0;

	///
	/// \brief numTriangles Get the number of triangles in the current node
	/// \return Number of triangles
	///
	virtual unsigned int numTriangles() const
	{
		return 0;
	}

	///
	/// \brief getBounds Get the AABB of the current node
	/// \return The aabb
	///
	AABB getBounds() const { return m_bounds; }

	///
	/// \brief nodeCount Recursively calculate the nodes beneath the current node
	/// \return The number of nodes that are children to this node
	///
	unsigned int nodeCount() const;

	///
	/// \brief surfaceArea Get the surface area of the nodes AABB
	/// \return
	///
	float surfaceArea() const;

	///
	/// \brief cleanUp Recursively free the allocated memory
	///
	void cleanUp();

protected:
	///
	/// \brief m_bounds AABB of the node
	///
	AABB m_bounds;
};

///
/// \brief The InnerNode class Inner node, e.g. a node with children
///
class InnerNode : public BVHNode
{
public:
	///
	/// \brief InnerNode Default dtor for the inner node, inner node consists of left and right children
	/// \param _bounds AABB of the node
	/// \param _leftChild Left child node of the inner node
	/// \param _rightChild Right child node of the inner node
	///
	InnerNode(const AABB &_bounds, BVHNode *_leftChild, BVHNode *_rightChild) :
		m_children{_leftChild, _rightChild}
	{
		m_bounds = _bounds;
	}

	///
	/// \brief isLeaf Inner node is never a leaf but needs to implement this to distinquish the abstract nodes from each other
	/// \return False
	///
	bool isLeaf() const override
	{
		return false;
	}

	///
	/// \brief numChildNodes Number of child nodes, an inner node will always have two child nodes; left and right
	/// \return 2
	///
	unsigned int numChildNodes() const override
	{
		return 2;
	}

	///
	/// \brief childNode Get a child node of the current node
	/// \param _index Index of the child wanted
	/// \return The child node
	///
	BVHNode* childNode(const unsigned int &_index) const override
	{
		// Sanity check for the index
		assert(_index < 2);
		return m_children[_index];
	}

private:
	///
	/// \brief m_children Pointers to the children of the this node
	///
	BVHNode *m_children[2];
};

///
/// \brief The LeafNode class Simple leaf node class, stores indices to triangles inside the node
///
class LeafNode : public BVHNode
{
public:
	///
	/// \brief LeafNode Default ctor for a leaf node
	/// \param _bounds AABB of the node
	/// \param _firstInd Index to the first triangle contained by this node
	/// \param _lastInd Index to the last triangle contained by this node
	///
	LeafNode(const AABB &_bounds, const unsigned int &_firstInd, const unsigned int &_lastInd) :
		m_firstTriIndex(_firstInd),
		m_lastTriIndex(_lastInd)
	{
		m_bounds = _bounds;
	}

	///
	/// \brief isLeaf Used to distinquish the abstract nodes from each other, will always return true
	/// \return True
	///
	bool isLeaf() const override
	{
		return true;
	}

	///
	/// \brief numChildNodes Leaf node has no children but needs to implement the virtual method
	/// \return 0
	///
	unsigned int numChildNodes() const override
	{
		return 0;
	}

	///
	/// \brief childNode Leaf node has no children but needs to implement the virtual method
	/// \param _index Index to the child
	/// \return nullptr as there are no children
	///
	BVHNode* childNode(const unsigned int &_index) const override
	{
		return nullptr;
	}

	///
	/// \brief numTriangles Get the number of triangles inside the leaf node
	/// \return The number of triangles contained by the node, e.g. last index - first index
	///
	unsigned int numTriangles() const override
	{
		return m_lastTriIndex - m_firstTriIndex;
	}

	///
	/// \brief firstIndex Get the first triangle index
	/// \return Index to the first triangle
	///
	unsigned int firstIndex() const { return m_firstTriIndex; }

	///
	/// \brief lastIndex Get the last triangle index
	/// \return Index to the last triangle
	///
	unsigned int lastIndex() const { return m_lastTriIndex; }

private:
	///
	/// \brief m_firstTriIndex Index of the first triangle
	///
	unsigned int m_firstTriIndex;

	///
	/// \brief m_lastTriIndex Index of the last triangle
	///
	unsigned int m_lastTriIndex;
};
