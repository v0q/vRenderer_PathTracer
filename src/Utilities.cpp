#include "Utilities.h"

namespace vUtilities
{
	vFloat3 minvFloat3(const vFloat3 &_a, const vFloat3 &_b)
	{
		return vFloat3(_a.x < _b.x ? _a.x : _b.x,
									 _a.y < _b.y ? _a.y : _b.y,
									 _a.z < _b.z ? _a.z : _b.z);
	}

	vFloat3 maxvFloat3(const vFloat3 &_a, const vFloat3 &_b)
	{
		return vFloat3(_a.x > _b.x ? _a.x : _b.x,
									 _a.y > _b.y ? _a.y : _b.y,
									 _a.z > _b.z ? _a.z : _b.z);
	}

	float dot(const vFloat3 &_a, const vFloat3 &_b)
	{
		return _a.x*_b.x + _a.y*_b.y + _a.z*_b.z;
	}

	vFloat3 cross(const vFloat3 &_a, const vFloat3 &_b)
	{
		return vFloat3(_a.y*_b.z - _a.z*_b.y, _a.z*_b.x - _a.x*_b.z, _a.x*_b.y - _a.y*_b.x);
	}
}
