#pragma once

#include <ngl/Vec3.h>

constexpr float kTriangleCost = 1.f;
constexpr float kNodeCost = 1.f;
constexpr unsigned int kMinLeafSize = 1;
constexpr unsigned int kSpatialBins = 32;
constexpr unsigned int kMaxDepth = 64;
constexpr unsigned int kMaxLeafSize = 64;

#include <iostream>
namespace vUtilities
{
	inline ngl::Vec3 minVec3(const ngl::Vec3 &_v1, const ngl::Vec3 &_v2)
	{
		return ngl::Vec3(_v1.m_x < _v2.m_x ? _v1.m_x : _v2.m_x,
										 _v1.m_y < _v2.m_y ? _v1.m_y : _v2.m_y,
										 _v1.m_z < _v2.m_z ? _v1.m_z : _v2.m_z);
	}

	inline ngl::Vec3 maxVec3(const ngl::Vec3 &_v1, const ngl::Vec3 &_v2)
	{
		return ngl::Vec3(_v1.m_x > _v2.m_x ? _v1.m_x : _v2.m_x,
										 _v1.m_y > _v2.m_y ? _v1.m_y : _v2.m_y,
										 _v1.m_z > _v2.m_z ? _v1.m_z : _v2.m_z);
	}

	inline ngl::Vec3 lerp(const ngl::Vec3 &_v1, const ngl::Vec3 &_v2, const float &_t)
	{
		return _v1 * (1.f - _t) + _v2 * _t;
	}

	inline float clamp(const float &_v, const float &_min, const float &_max)
	{
		return std::max(_min, std::min(_v, _max));
	}

	inline void clampVec3(ngl::Vec3 &io_v, const ngl::Vec3 &_low, const ngl::Vec3 &_high)
	{
		io_v.m_x = vUtilities::clamp(io_v.m_x, _low.m_x, _high.m_x);
		io_v.m_y = vUtilities::clamp(io_v.m_y, _low.m_y, _high.m_y);
		io_v.m_z = vUtilities::clamp(io_v.m_z, _low.m_z, _high.m_z);
	}

	template <class T> inline void swap(T &_a, T &_b)
	{
		T tmp = _a;
		_a = _b;
		_b = tmp;
	}
}
