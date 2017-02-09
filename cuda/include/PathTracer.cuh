#pragma once

typedef struct vVert {
	float4 m_vert;
	float4 m_normal;
} vVert;

typedef struct vBoundingBox {
	float2 m_x;
	float2 m_y;
	float2 m_z;
} vBoundingBox;

typedef struct vTriangle {
	vVert m_v1;
	vVert m_v2;
	vVert m_v3;
} vTriangle;

typedef struct vMesh {
	vTriangle *m_mesh;
	vBoundingBox m_bb;
	unsigned int m_triCount;
} vMesh;

typedef struct vHitData {
	float4 m_hitPoint;
	float4 m_normal;
	float4 m_emission;
	float4 m_color;
} vHitData;

void cu_runRenderKernel(cudaSurfaceObject_t _tex, const vMesh *_scene, float4 *_colors, float4 *_cam, float4 *_dir, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time);
void cu_fillFloat4(float4 *d_ptr, float4 _val, unsigned int _size);
