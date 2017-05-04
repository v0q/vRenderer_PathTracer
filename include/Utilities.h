///
/// \file Utilities.h
/// \brief Simple utility functions used by the SBVH builder
/// \authors Teemu Lindborg
/// \version 1.0
/// \date 04/05/17 Updated to NCCA Coding standard
/// Revision History :
/// Initial Version 08/12/16
/// \todo -
///

#pragma once

#include <ngl/Vec3.h>

constexpr float kTriangleCost = 1.f;
constexpr float kNodeCost = 1.f;
constexpr unsigned int kMinLeafSize = 4;
constexpr unsigned int kSpatialBins = 32;
constexpr unsigned int kMaxDepth = 64;
constexpr unsigned int kMaxLeafSize = 64;

///
/// \brief Utility namespace
///
namespace vUtilities
{
	///
	/// \brief minVec3 Get the component-wise minimums from two vectors
	/// \param _v1 First vector
	/// \param _v2 Second vector
	/// \return Vector with minimum components of the two given vectors
	///
	inline ngl::Vec3 minVec3(const ngl::Vec3 &_v1, const ngl::Vec3 &_v2)
	{
		return ngl::Vec3(_v1.m_x < _v2.m_x ? _v1.m_x : _v2.m_x,
										 _v1.m_y < _v2.m_y ? _v1.m_y : _v2.m_y,
										 _v1.m_z < _v2.m_z ? _v1.m_z : _v2.m_z);
	}

	///
	/// \brief maxVec3 Get the component-wise maximums from two vectors
	/// \param _v1 First vector
	/// \param _v2 Second vector
	/// \return Vector with maximum components of the two given vectors
	///
	inline ngl::Vec3 maxVec3(const ngl::Vec3 &_v1, const ngl::Vec3 &_v2)
	{
		return ngl::Vec3(_v1.m_x > _v2.m_x ? _v1.m_x : _v2.m_x,
										 _v1.m_y > _v2.m_y ? _v1.m_y : _v2.m_y,
										 _v1.m_z > _v2.m_z ? _v1.m_z : _v2.m_z);
	}

	///
	/// \brief lerp Perform linear interpolation between two vectors given an alpha
	/// \param _v1 First vector
	/// \param _v2 Second vector
	/// \param _t Alpha
	/// \return Interpolated vector _v1 * (1.f - _t) + _v2 * _t
	///
	inline ngl::Vec3 lerp(const ngl::Vec3 &_v1, const ngl::Vec3 &_v2, const float &_t)
	{
		return _v1 * (1.f - _t) + _v2 * _t;
	}

	///
	/// \brief clamp Clamps a given float between two values
	/// \param _v Value to clamp
	/// \param _min Minimum limit of the clamp
	/// \param _max Maximum limit of the clamp
	/// \return Clamped value
	///
	inline float clamp(const float &_v, const float &_min, const float &_max)
	{
		return std::max(_min, std::min(_v, _max));
	}

	///
	/// \brief clampVec3 Performs component-wise clamping for a vector
	/// \param io_v Input vector to be clamped
	/// \param _low Minimum limit of the clamp
	/// \param _high Maximum limit of the clamp
	///
	inline void clampVec3(ngl::Vec3 &io_v, const ngl::Vec3 &_low, const ngl::Vec3 &_high)
	{
		io_v.m_x = vUtilities::clamp(io_v.m_x, _low.m_x, _high.m_x);
		io_v.m_y = vUtilities::clamp(io_v.m_y, _low.m_y, _high.m_y);
		io_v.m_z = vUtilities::clamp(io_v.m_z, _low.m_z, _high.m_z);
	}

	///
	/// \brief swap Swaps two elements around
	/// \param _a Element to be swapped with _b
	/// \param _b Element to be swapped with _a
	///
	template<typename T> inline void swap(T &_a, T &_b)
	{
		T tmp = _a;
		_a = _b;
		_b = tmp;
	}
}
