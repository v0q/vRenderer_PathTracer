#pragma once

#include "MeshData.h"

namespace vUtilities
{
	vFloat3 minvFloat3(const vFloat3 &_a, const vFloat3 &_b);
	vFloat3 maxvFloat3(const vFloat3 &_a, const vFloat3 &_b);
	float dot(const vFloat3 &_a, const vFloat3 &_b);
	vFloat3 cross(const vFloat3 &_a, const vFloat3 &_b);
}
