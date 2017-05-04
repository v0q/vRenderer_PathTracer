///
/// \file AABB.h
/// \brief Simple AABB used with the SBVH
/// \authors Teemu Lindborg
/// \version 1.0
/// \date 04/05/17 Updated to NCCA Coding standard
/// Revision History :
/// Initial Version 08/12/16
/// \todo -
///

#pragma once

#include <cfloat>
#include <ngl/Vec3.h>
#include <iostream>

#include "Utilities.h"

///
/// \brief The AABB class, encapsulates a simple AABB class to be used with building the acceleration structure for the path tracer
///
class AABB
{
public:
	///
	/// \brief AABB Default Ctor, initialises the bounding box to FLT_MAX
	///
	AABB() :
		m_min(FLT_MAX, FLT_MAX, FLT_MAX),
		m_max(-FLT_MAX, -FLT_MAX, -FLT_MAX)
	{}

	///
	/// \brief AABB Ctor using given min and max bounds
	/// \param _min Minimum bounds
	/// \param _max Maximum bounds
	///
	AABB(const ngl::Vec3 &_min, const ngl::Vec3 &_max) :
		m_min(_min),
		m_max(_max)
	{}

	///
	/// \brief extendBB Extends the bounding box to cover the given point
	/// \param _vert Point to used to extend the bounding box
	///
	inline void extendBB(const ngl::Vec3 &_vert)
	{
		m_min = vUtilities::minVec3(m_min, _vert);
		m_max = vUtilities::maxVec3(m_max, _vert);
	}

	///
	/// \brief extendBB Extends the bounding box to cover the another bounding box
	/// \param _aabb AABB to cover
	///
	inline void extendBB(const AABB &_aabb)
	{
		m_min = vUtilities::minVec3(m_min, _aabb.m_min);
		m_max = vUtilities::maxVec3(m_max, _aabb.m_max);
	}

	///
	/// \brief intersectBB Get the intersection between two bounding boxes
	/// \param _aabb Bounding box to check the intersection against
	///
	inline void intersectBB(const AABB &_aabb)
	{
		m_min = vUtilities::maxVec3(m_min, _aabb.minBounds());
		m_max = vUtilities::minVec3(m_max, _aabb.maxBounds());
	}

	///
	/// \brief getSurfaceArea Calculates the surface area of the AABB
	/// \return The surface area
	///
	inline float surfaceArea() const
	{
		float lenX = m_max.m_x - m_min.m_x;
		float lenY = m_max.m_y - m_min.m_y;
		float lenZ = m_max.m_z - m_min.m_z;

		return 2*(lenX*lenY + lenY*lenZ + lenZ*lenX);
	}

	///
	/// \brief setMinBoundsComponent Update a single component (X, Y, Z) of the minimum bounds
	/// \param _e Index of the component 0...3 (x..z)
	/// \param _val New value for the component
	///
	inline void setMinBoundsComponent(const unsigned int &_e, const float &_val)
	{
		m_min.m_openGL[_e] = _val;
	}

	///
	/// \brief setMaxBoundsComponent Update a single component (X, Y, Z) of the maximum bounds
	/// \param _e Index of the component 0...3 (x..z)
	/// \param _val New value for the component
	///
	inline void setMaxBoundsComponent(const unsigned int &_e, const float &_val)
	{
		m_max.m_openGL[_e] = _val;
	}

	///
	/// \brief isValid Checks whether the bounds of the AABB are valid, e.g. if mininimum bounds < maximum bounds
	/// \return Whether the AABB is valid or not
	///
	inline bool isValid() const
	{
		return (m_min.m_x < m_max.m_x && m_min.m_y < m_max.m_y && m_min.m_z < m_max.m_z);
	}

	///
	/// \brief minBounds Get the minimum bounds of the AABB
	/// \return The minimum bounds
	///
	inline ngl::Vec3 minBounds() const
	{
		return m_min;
	}

	///
	/// \brief maxBounds Get the maximum bounds of the AABB
	/// \return The maximum bounds
	///
	inline ngl::Vec3 maxBounds() const
	{
		return m_max;
	}

	///
	/// \brief getCentroid Get the centroid point of the AABB
	/// \return The centroid point
	///
	inline ngl::Vec3 getCentroid() const
	{
		return (m_min + m_max) / 2.f;
	}

	///
	/// \brief printBounds Prints the minimum and maximum bounds of the AABB, used for debugging
	///
	inline void printBounds() const
	{
		std::cout << "AABB bounds:\n";
		std::cout << "  Min: [" << m_min.m_x << ", " << m_min.m_y << ", " << m_min.m_z << "]\n";
		std::cout << "  Max: [" << m_max.m_x << ", " << m_max.m_y << ", " << m_max.m_z << "]\n";
	}

private:
	///
	/// \brief m_min Minimum bounds
	///
	ngl::Vec3 m_min;

	///
	/// \brief m_max Maximum bounds
	///
	ngl::Vec3 m_max;
};
