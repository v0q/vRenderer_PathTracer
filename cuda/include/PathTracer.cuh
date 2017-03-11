#pragma once

typedef struct vHitData {
	float4 m_hitPoint;
	float4 m_normal;
	float4 m_emission;
	float4 m_color;
} vHitData;

void cu_runRenderKernel(cudaSurfaceObject_t _texture, float4 *_hdr, float4 *_vertices, float4 *_normals, float4 *_bvhData, unsigned int *_triIdxList,
												unsigned int _vertCount, unsigned int _bvhNodeCount, unsigned int _triIdxCount,
												float4 *_colorArr, float4 *_cam, float4 *_dir,
												unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time);
void cu_setHDRDim(const unsigned int &_w, const unsigned int &_h);
void cu_fillFloat4(float4 *d_ptr, float4 _val, unsigned int _size);
