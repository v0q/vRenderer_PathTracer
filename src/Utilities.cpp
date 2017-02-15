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

#include <assert.h>

#include "Utilities.h"

namespace vUtilities
{
	/// The following section is from :-
	/// Sam Lapere (September 20, 2016). GPU path tracing tutorial 4: Optimised BVH building, faster traversal and intersection kernels and HDR environment lighting [online].
	/// [Accessed 2017]. Available from: http://raytracey.blogspot.co.uk/2016/09/gpu-path-tracing-tutorial-4-optimised.html & https://github.com/straaljager/GPU-path-tracing-tutorial-4
	#define QSORT_STACK_SIZE    32

	static inline void InsertionSort(const unsigned int &_start, const unsigned int &_size, void* io_data, const SortCompareFunc &_compareFunc, const SortSwapFunc &_swapFunc);
	static inline int Median3(int io_low, int _high, void* io_data, const SortCompareFunc &_compareFunc);
	static void Qsort(int io_low, int io_high, void* io_data, const SortCompareFunc &_compareFunc, const SortSwapFunc &_swapFunc);

	void InsertionSort(const unsigned int &_start, const unsigned int &_size, void *io_data, const SortCompareFunc &_compareFunc, const SortSwapFunc &_swapFunc)
	{
		assert(_compareFunc && _swapFunc);

		for(unsigned int i = 1; i < _size; ++i)
		{
			unsigned int j = _start + i - 1;
			while(j >= _start && _compareFunc(io_data, j, j + 1) > 0)
			{
				_swapFunc(io_data, j, j + 1);
				--j;
			}
		}
	}

	int Median3(int _low, int _high, void *io_data, const SortCompareFunc &_compareFunc)
	{
		assert(_compareFunc);
//		assert(_high >= 2);

		int l = _low;
		int c = (_low + _high) >> 1;
		int h = _high - 2;

		if(_compareFunc(io_data, l, h) > 0)
			swap(l, h);
		if(_compareFunc(io_data, l, c) > 0)
			c = l;

		return (_compareFunc(io_data, c, h) > 0) ? h : c;
	}

	void Qsort(int io_low, int io_high, void *io_data, const SortCompareFunc &_compareFunc, const SortSwapFunc &_swapFunc)
	{
		assert(_compareFunc && _swapFunc);
//		assert(io_low <= io_high);

		int stack[QSORT_STACK_SIZE];
		int sp = 0;
		stack[sp++] = io_high;

		while(sp)
		{
			io_high = stack[--sp];
//			assert(io_low <= io_high);

			// Use insertion sort for small values or if stack gets full.
			if(io_high - io_low <= 15 || sp + 2 > QSORT_STACK_SIZE)
			{
				InsertionSort(io_low, io_high - io_low, io_data, _compareFunc, _swapFunc);
				io_low = io_high + 1;
				continue;
			}

			// Select pivot using median-3, and hide it in the highest entry.
			_swapFunc(io_data, Median3(io_low, io_high, io_data, _compareFunc), io_high - 1);

			// Partition data.
			int i = io_low - 1;
			int j = io_high - 1;
			for(;;)
			{
				do
				{
					i++;
				} while(_compareFunc(io_data, i, io_high - 1) < 0);
				do
				{
					j--;
				}	while(_compareFunc(io_data, j, io_high - 1) > 0);

//				assert(i >= io_low && j >= io_low && i < io_high && j < io_high);
				if(i >= j)
					break;

				_swapFunc(io_data, i, j);
			}

			// Restore pivot.
			_swapFunc(io_data, i, io_high - 1);

			// Sort sub-partitions.
//			assert(sp + 2 <= QSORT_STACK_SIZE);
			if(io_high - i > 2)
				stack[sp++] = io_high;
			if(i - io_low > 1)
				stack[sp++] = i;
			else
				io_low = i + 1;
		}
	}

	void Sort(int _start, int _end, void *io_data, const SortCompareFunc &_compareFunc, const SortSwapFunc &_swapFunc)
	{
//		assert(_start <= _end);
		assert(_compareFunc && _swapFunc);

		if(_start + 2 <= _end)
			Qsort(_start, _end, io_data, _compareFunc, _swapFunc);
	}

	int CompareInt(void *io_data, const unsigned int &_idxA, const unsigned int &_idxB)
	{
		int a = ((int*)io_data)[_idxA];
		int b = ((int*)io_data)[_idxB];
		return (a < b) ? -1 : (a > b) ? 1 : 0;
	}

	void SwapInt(void *io_data, const unsigned int &_idxA, const unsigned int &_idxB)
	{
		swap(((int*)io_data)[_idxA], ((int*)io_data)[_idxB]);
	}

	int CompareFloat(void *io_data, const unsigned int &_idxA, const unsigned int &_idxB)
	{
		float a = ((float*)io_data)[_idxA];
		float b = ((float*)io_data)[_idxB];
		return (a < b) ? -1 : (a > b) ? 1 : 0;
	}

	void SwapFloat(void *io_data, const unsigned int &_idxA, const unsigned int &_idxB)
	{
		swap(((float*)io_data)[_idxA], ((float*)io_data)[_idxB]);
	}
}
