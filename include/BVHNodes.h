#pragma once

#include <assert.h>
#include "AABB.h"

class BVHNode
{
public:
	BVHNode() {}
	virtual ~BVHNode() {}

	virtual BVHNode* childNode(const unsigned int &_index) const = 0;
	virtual bool isLeaf() const = 0;
	virtual unsigned int numChildNodes() const = 0;

	virtual unsigned int numTriangles() const
	{
		return 0;
	}

	AABB getBounds() const { return m_bounds; }
	unsigned int nodeCount() const;
	float surfaceArea() const;
	void cleanUp();

protected:
	AABB m_bounds;
};

class InnerNode : public BVHNode
{
public:
	InnerNode(const AABB &_bounds, BVHNode *_leftChild, BVHNode *_rightChild) :
		m_children{_leftChild, _rightChild}
	{
		m_bounds = _bounds;
	}

	bool isLeaf() const override
	{
		return false;
	}

	unsigned int numChildNodes() const override
	{
		return 2;
	}

	BVHNode* childNode(const unsigned int &_index) const override
	{
		assert(_index < 2);
		return m_children[_index];
	}

private:
	BVHNode *m_children[2];
};

class LeafNode : public BVHNode
{
public:
	LeafNode(const AABB &_bounds, const unsigned int &_firstInd, const unsigned int &_lastInd) :
		m_firstTriIndex(_firstInd),
		m_lastTriIndex(_lastInd)
	{
		m_bounds = _bounds;
	}

	bool isLeaf() const override
	{
		return true;
	}

	unsigned int numChildNodes() const override
	{
		return 0;
	}

	BVHNode* childNode(const unsigned int &_index) const override
	{
		return nullptr;
	}

	unsigned int numTriangles() const override
	{
		return m_lastTriIndex - m_firstTriIndex;
	}

	unsigned int firstIndex() const { return m_firstTriIndex; }
	unsigned int lastIndex() const { return m_lastTriIndex; }

private:
	unsigned int m_firstTriIndex;
	unsigned int m_lastTriIndex;
};
