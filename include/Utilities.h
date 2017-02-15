#pragma once

#include "vDataTypes.h"

namespace vUtilities
{
	inline vFloat3 minvFloat3(const vFloat3 &_a, const vFloat3 &_b)
	{
		return vFloat3(_a.x < _b.x ? _a.x : _b.x,
									 _a.y < _b.y ? _a.y : _b.y,
									 _a.z < _b.z ? _a.z : _b.z);
	}

	inline vFloat3 maxvFloat3(const vFloat3 &_a, const vFloat3 &_b)
	{
		return vFloat3(_a.x > _b.x ? _a.x : _b.x,
									 _a.y > _b.y ? _a.y : _b.y,
									 _a.z > _b.z ? _a.z : _b.z);
	}

	inline float min3f(const float &_a, const float &_b, const float &_c)
	{
		return std::min(std::min(_a, _b), _c);
	}

	inline vInt3 clamp(const vInt3 &_a, const vInt3 &_b, const vInt3 &_c)
	{
		return vInt3(std::max(std::min(_a.x, _c.x), _b.x),
								 std::max(std::min(_a.y, _c.y), _b.y),
								 std::max(std::min(_a.z, _c.z), _b.z));
	}

	inline vFloat3 clamp(const vFloat3 &_a, const vFloat3 &_b, const vFloat3 &_c)
	{
		return vFloat3(std::max(std::min(_a.x, _c.x), _b.x),
									 std::max(std::min(_a.y, _c.y), _b.y),
									 std::max(std::min(_a.z, _c.z), _b.z));
	}

	inline float dot(const vFloat3 &_a, const vFloat3 &_b)
	{
		return _a.x*_b.x + _a.y*_b.y + _a.z*_b.z;
	}

	inline vFloat3 cross(const vFloat3 &_a, const vFloat3 &_b)
	{
		return vFloat3(_a.y*_b.z - _a.z*_b.y, _a.z*_b.x - _a.x*_b.z, _a.x*_b.y - _a.y*_b.x);
	}

	/// The following section is from :-
	/// Sam Lapere (September 20, 2016). GPU path tracing tutorial 4: Optimised BVH building, faster traversal and intersection kernels and HDR environment lighting [online].
	/// [Accessed 2017]. Available from: http://raytracey.blogspot.co.uk/2016/09/gpu-path-tracing-tutorial-4-optimised.html & https://github.com/straaljager/GPU-path-tracing-tutorial-4
	typedef int(*SortCompareFunc) (void *io_data, const unsigned int &_idxA, const unsigned int &_idxB);
	typedef void(*SortSwapFunc) (void *io_data, const unsigned int &_idxA, const unsigned int &_idxB);

	void Sort(int _start, int _end, void *io_data, const SortCompareFunc &_compareFunc, const SortSwapFunc &_swapFunc);

	int CompareInt(void *io_data, const unsigned int &_idxA, const unsigned int &_idxB);
	void SwapInt(void *io_data, const unsigned int &_idxA, const unsigned int &_idxB);

	int CompareFloat(void *io_data, const unsigned int &_idxA, const unsigned int &_idxB);
	void SwapFloat(void *io_data, const unsigned int &_idxA, const unsigned int &_idxB);

	template <class T> inline void swap(T& a, T& b)
	{
		T tmp = a;
		a = b;
		b = tmp;
	}

	template <class A> inline A clamp(const A& a, const A& b, const A& c)
	{
		return std::max(std::min(a, c), b);
	}

	template <class A, class B> inline A lerp(const A& a, const A& b, const B& t)
	{
		return (A)(a * ((B)1 - t) + b * t);
	}
	/// end of Citation
}
