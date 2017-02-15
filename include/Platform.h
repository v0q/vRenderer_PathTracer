/*
*  Copyright (c) 2009-2011, NVIDIA Corporation
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions are met:
*      * Redistributions of source code must retain the above copyright
*        notice, this list of conditions and the following disclaimer.
*      * Redistributions in binary form must reproduce the above copyright
*        notice, this list of conditions and the following disclaimer in the
*        documentation and/or other materials provided with the distribution.
*      * Neither the name of NVIDIA Corporation nor the
*        names of its contributors may be used to endorse or promote products
*        derived from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
*  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
*  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
*  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
*  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
*  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
*  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

///
/// @brief Platform
/// Adapted from :-
/// Sam Lapere (January 18, 2016). GPU path tracing tutorial 3: GPU-friendly Acceleration Structures. Now you're cooking with GAS! [online].
/// [Accessed 2017]. Available from: http://raytracey.blogspot.co.uk/2016/01/gpu-path-tracing-tutorial-3-take-your.html & https://github.com/straaljager/GPU-path-tracing-tutorial-3/
///
/// Reformatted the original code to match the code layout and my implementation
///
/// @author Teemu Lindborg
/// @version 0.1
/// @date 15/01/17 Initial version
/// Revision History :
/// -
/// @todo Tidying up and rewriting
///

#include <iostream>

class Platform
{
public:
	///
	/// \brief Platform Default ctor
	///
	Platform()
	{
		m_name = std::string("Default");
		m_SAHNodeCost = 1.f;
		m_SAHTriangleCost = 1.f;
		m_triBatchSize = 1;
		m_nodeBatchSize = 1;
		m_minLeafSize = 1;
		m_maxLeafSize = 0x7FFFFFF;
	}

	///
	/// \brief Platform Default ctor
	/// \param _name Name of the platform
	/// \param _nodeCost
	/// \param _triCost
	/// \param _nodeBatchSize
	/// \param _triBatchSize
	///
	Platform(const std::string &_name, const float &_nodeCost = 1.f, const float &_triCost = 1.f, const unsigned int &_nodeBatchSize = 1, const unsigned int &_triBatchSize = 1) :
		m_name(_name),
		m_SAHNodeCost(_nodeCost),
		m_SAHTriangleCost(_triCost),
		m_triBatchSize(_triBatchSize),
		m_nodeBatchSize(_nodeBatchSize),
		m_minLeafSize(1),
		m_maxLeafSize(0x7FFFFFF)
	{ }

	///
	/// \brief getName Get the name of the platform
	/// \return The name of the platform
	///
	const std::string& getName() const
	{
		return m_name;
	}

	// SAH weights
	///
	/// \brief getSAHTriangleCost
	/// \return
	///
	float getSAHTriangleCost() const
	{
		return m_SAHTriangleCost;
	}

	///
	/// \brief getSAHNodeCost
	/// \return
	///
	float getSAHNodeCost() const
	{
		return m_SAHNodeCost;
	}

	// SAH costs, raw and batched
	///
	/// \brief getCost
	/// \param _numChildNodes
	/// \param _numTris
	/// \return
	///
	float getCost(const unsigned int &_numChildNodes, const unsigned int &_numTris) const
	{
		return getNodeCost(_numChildNodes) + getTriangleCost(_numTris);
	}

	///
	/// \brief getTriangleCost
	/// \param _n
	/// \return
	///
	float getTriangleCost(const unsigned int &_n) const
	{
		return roundToTriangleBatchSize(_n) * m_SAHTriangleCost;
	}

	///
	/// \brief getNodeCost
	/// \param _n
	/// \return
	///
	float getNodeCost(const unsigned int &_n) const
	{
		return roundToNodeBatchSize(_n) * m_SAHNodeCost;
	}

	// batch processing (how many ops at the price of one)
	///
	/// \brief getTriangleBatchSize
	/// \return
	///
	unsigned int getTriangleBatchSize() const
	{
		return m_triBatchSize;
	}

	///
	/// \brief getNodeBatchSize
	/// \return
	///
	unsigned int getNodeBatchSize() const
	{
		return m_nodeBatchSize;
	}

	///
	/// \brief setTriangleBatchSize
	/// \param _triBatchSize
	///
	void setTriangleBatchSize(const unsigned int &_triBatchSize)
	{
		m_triBatchSize = _triBatchSize;
	}

	///
	/// \brief setNodeBatchSize
	/// \param _nodeBatchSize
	///
	void setNodeBatchSize(const unsigned int &_nodeBatchSize)
	{
		m_nodeBatchSize = _nodeBatchSize;
	}

	///
	/// \brief roundToTriangleBatchSize
	/// \param _n
	/// \return
	///
	unsigned int roundToTriangleBatchSize(const unsigned int &_n) const
	{
		return ((_n + m_triBatchSize - 1) / m_triBatchSize)*m_triBatchSize;
	}

	///
	/// \brief roundToNodeBatchSize
	/// \param _n
	/// \return
	///
	unsigned int roundToNodeBatchSize(const int &_n) const
	{
		return ((_n + m_nodeBatchSize - 1) / m_nodeBatchSize)*m_nodeBatchSize;
	}

	// leaf preferences
	///
	/// \brief setLeafPreferences
	/// \param _minSize
	/// \param _maxSize
	///
	void setLeafPreferences(const unsigned int &_minSize, const unsigned int &_maxSize)
	{
		m_minLeafSize = _minSize;
		m_maxLeafSize = _maxSize;
	}

	///
	/// \brief getMinLeafSize
	/// \return
	///
	unsigned int getMinLeafSize() const
	{
		return m_minLeafSize;
	}

	///
	/// \brief getMaxLeafSize
	/// \return
	///
	unsigned int getMaxLeafSize() const
	{
		return m_maxLeafSize;
	}

private:
	///
	/// \brief m_name
	///
	std::string m_name;

	///
	/// \brief m_SAHNodeCost
	///
	float m_SAHNodeCost;

	///
	/// \brief m_SAHTriangleCost
	///
	float m_SAHTriangleCost;

	///
	/// \brief m_triBatchSize
	///
	unsigned int m_triBatchSize;

	///
	/// \brief m_nodeBatchSize
	///
	unsigned int m_nodeBatchSize;

	///
	/// \brief m_minLeafSize
	///
	unsigned int m_minLeafSize;

	///
	/// \brief m_maxLeafSize
	///
	unsigned int m_maxLeafSize;
};
