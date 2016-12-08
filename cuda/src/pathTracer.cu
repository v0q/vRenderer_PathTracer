#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "pathTracer.cuh"

surface<void, 2> textureOut;

__global__ void cuda_T(cudaSurfaceObject_t tex, unsigned int w, unsigned int h, float c)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < w || y < h) {
		unsigned char r = (unsigned char)(c*255);
		unsigned char g = (unsigned char)((x/(float)w) * 255);
		unsigned char b = (unsigned char)((y/(float)h) * 255);
		uchar4 data = make_uchar4(r, g, b, 0xff);
//		float4 data = make_float4(1.0, 1.0, 1.0, 1.0);
		surf2Dwrite(data, tex, x*sizeof(uchar4), y);
	}
}

void cu_ModifyTexture(cudaSurfaceObject_t _texture, unsigned int _w, unsigned int _h, float c)
{
	dim3 dimBlock(16, 16);
	dim3 dimGrid((_w / dimBlock.x),
							 (_h / dimBlock.y));

	cuda_T<<<dimGrid, dimBlock>>>(_texture, _w, _h, c);

//	std::cout << _w << " " << _h << "\n";
}
