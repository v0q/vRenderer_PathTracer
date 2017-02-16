#pragma once

#include <ngl/Vec3.h>

constexpr float kTriangleCost = 1.f;
constexpr float kNodeCost = 1.f;
constexpr unsigned int kMinLeafSize = 1;
constexpr unsigned int kMaxDepth = 64;

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

	template <class T> inline void swap(T &_a, T &_b)
	{
		T tmp = _a;
		_a = _b;
		_b = tmp;
	}
}
