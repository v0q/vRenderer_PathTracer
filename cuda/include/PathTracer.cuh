#pragma once

typedef struct vHitData {
	float4 m_hitPoint;
	float4 m_normal;
	float4 m_emission;
	float4 m_color;
	unsigned int m_hitType;
} vHitData;

typedef struct vCamera {
	float4 m_origin;
	float4 m_dir;
	float4 m_upV;
	float4 m_rightV;
	float m_fovScale;
} vCamera;

void cu_runRenderKernel(cudaSurfaceObject_t _texture, cudaSurfaceObject_t _depth,
												float4 *_hdr, float4 *_vertices, float4 *_normals, float4 *_bvhData, unsigned int *_triIdxList,
												unsigned int _vertCount, unsigned int _bvhNodeCount, unsigned int _triIdxCount,
												float4 *_colorArr, vCamera _cam, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time);
void cu_loadDiffuse(const float4 *_diffuse, const unsigned int _w, const unsigned int _h);
void cu_setHDRDim(const unsigned int &_w, const unsigned int &_h);
void cu_fillFloat4(float4 *d_ptr, float4 _val, unsigned int _size);
