#pragma once

#include <cfloat>
#include <ngl/Vec3.h>

#include "Utilities.h"

class AABB
{
public:
	AABB() :
		m_min(FLT_MAX, FLT_MAX, FLT_MAX),
		m_max(-FLT_MAX, -FLT_MAX, -FLT_MAX)
	{}

	AABB(const ngl::Vec3 &_min, const ngl::Vec3 &_max) :
		m_min(_min),
		m_max(_max)
	{}

	///
	/// \brief extendBB
	/// \param _vert
	///
	inline void extendBB(const ngl::Vec3 &_vert)
	{
		m_min = vUtilities::minVec3(m_min, _vert);
		m_max = vUtilities::maxVec3(m_max, _vert);
	}

	///
	/// \brief extendBB
	/// \param _aabb
	///
	inline void extendBB(const AABB &_aabb)
	{
		m_min = vUtilities::minVec3(m_min, _aabb.m_min);
		m_max = vUtilities::maxVec3(m_max, _aabb.m_max);
	}

	///
	/// \brief intersectBB
	/// \param _aabb
	///
	inline void intersectBB(const AABB &_aabb)
	{
		m_min = vUtilities::maxVec3(m_min, _aabb.maxBounds());
		m_max = vUtilities::maxVec3(m_max, _aabb.minBounds());
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
	/// \brief setMinBoundsComponent
	/// \param _e
	/// \param _val
	///
	inline void setMinBoundsComponent(const unsigned int &_e, const float &_val)
	{
		m_min.m_openGL[_e] = _val;
	}

	///
	/// \brief setMaxBoundsComponent
	/// \param _e
	/// \param _val
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

	inline ngl::Vec3 minBounds() const
	{
		return m_min;
	}

	inline ngl::Vec3 maxBounds() const
	{
		return m_max;
	}

	inline ngl::Vec3 getCentroid() const
	{
		return (m_min + m_max) / 2.f;
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
