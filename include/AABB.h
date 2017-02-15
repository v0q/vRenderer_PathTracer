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
/// @brief Simple AABB class to be used with the SBVH
/// Adapted from :-
/// Sam Lapere (September 20, 2016). GPU path tracing tutorial 4: Optimised BVH building, faster traversal and intersection kernels and HDR environment lighting [online].
/// [Accessed 2017]. Available from: http://raytracey.blogspot.co.uk/2016/09/gpu-path-tracing-tutorial-4-optimised.html & https://github.com/straaljager/GPU-path-tracing-tutorial-4
///
/// Reformatted the original code to match the code layout and my implementation
///
/// @author Teemu Lindborg
/// @version 0.1
/// @date 15/01/17 Initial version
/// Revision History :
/// -
/// @todo -
///

#include <cfloat>

#include "vDataTypes.h"
#include "Utilities.h"

class AABB
{
public:
	///
	/// \brief AABB Default ctor, initialises the bounds to FLT_MAX and -FLT_MAX
	///
	inline AABB() :
		m_min(FLT_MAX, FLT_MAX, FLT_MAX),
		m_max(-FLT_MAX, -FLT_MAX, -FLT_MAX)
	{}

	///
	/// \brief AABB Ctor with defined bounds
	/// \param _min Minimum bounds of the AABB
	/// \param _max Maximum bounds of the AABB
	///
	inline AABB(const vFloat3 &_min, const vFloat3 &_max) :
		m_min(_min),
		m_max(_max)
	{}

	///
	/// \brief grow Grows the min and max bounds to contain a given point
	/// \param _v Point to grow the AABB with
	///
	inline void grow(const vFloat3 &_v)
	{
		m_min = vUtilities::minvFloat3(m_min, _v);
		m_max = vUtilities::maxvFloat3(m_min, _v);
	} // grows bounds to include 3d point pt

	///
	/// \brief grow Grow the current AABB to contain a given AABB as well
	/// \param _aabb The bounding box to grow the current one with
	///
	inline void grow(const AABB &_aabb)
	{
		grow(_aabb.m_min);
		grow(_aabb.m_max);
	}

	///
	/// \brief intersect Sets the current AABB to be the intersection of itself and another AABB
	/// \param _aabb AABB to intersect with
	///
	inline void intersect(const AABB &_aabb)
	{
		m_min = vUtilities::maxvFloat3(m_min, _aabb.m_min);
		m_max = vUtilities::minvFloat3(m_max, _aabb.m_max);
	}

	///
	/// \brief volume AABB side along X-axis * side along Y * side along Z
	/// \return the volume of the AABB
	///
	inline float volume() const
	{
		if(!valid())
			return 0.0f;
		return (m_max.x - m_min.x) * (m_max.y - m_min.y) * (m_max.z - m_min.z);
	}

	///
	/// \brief area Calculates the total area of the planes defining the AABB
	/// \return The calculated area
	///
	inline float area() const
	{
		if(!valid())
			return 0.0f;
		vFloat3 d = m_max - m_min;
		return (d.x*d.y + d.y*d.z + d.z*d.x)*2.0f;
	}

	///
	/// \brief valid Checks whether the current AABB is valid, e.g. if min bounds < max bounds
	/// \return True if valid
	///
	inline bool valid() const
	{
		return m_min.x <= m_max.x && m_min.y <= m_max.y && m_min.z <= m_max.z;
	}

	///
	/// \brief midPoint Get the midpoint of the AABB
	/// \return The mid point
	///
	inline vFloat3 midPoint() const
	{
		return (m_min + m_max)*0.5f;
	}

	///
	/// \brief minBounds Get the minimum dimensions of the AABB
	/// \return Minimum dimensions of the AABB
	///
	inline const vFloat3& minBounds() const
	{
		return m_min;
	}

	///
	/// \brief maxBounds Get the maximum dimensions of the AABB
	/// \return Maximum dimensions of the AABB
	///
	inline const vFloat3& maxBounds() const
	{
		return m_max;
	}

	inline void setMinBoundsComponent(const unsigned int &_ind, const float &_val)
	{
		m_min.v[_ind] = _val;
	}

	inline void setMaxBoundsComponent(const unsigned int &_ind, const float &_val)
	{
		m_max.v[_ind] = _val;
	}

	///
	/// \brief operator + Calculate the bounding box containing both of the given AABB's
	/// \param _aabb AABB to add
	/// \return The bounding box containing both the AABB's
	///
	inline AABB operator+(const AABB &_aabb) const
	{
		AABB u(*this);
		u.grow(_aabb);
		return u;
	}
//	inline vFloat3& min() { return m_min; }
//	inline vFloat3& max() { return m_max; }

private:
	///
	/// \brief m_min Minimum bounds
	///
	vFloat3 m_min;

	///
	/// \brief m_max Maximum bounds
	///
	vFloat3 m_max; // Max bounds
};
