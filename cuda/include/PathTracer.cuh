#pragma once

typedef struct vHitData {
	float4 m_hitPoint;
	float4 m_normal;
	float4 m_emission;
	float4 m_color;
	float4 m_specularColor;
	unsigned int m_hitType;
} vHitData;

typedef struct vCamera {
	float4 m_origin;
	float4 m_dir;
	float4 m_upV;
	float4 m_rightV;
	float m_fovScale;
} vCamera;

typedef enum vTextureType { DIFFUSE, NORMAL, SPECULAR } vTextureType;

void cu_runRenderKernel(cudaSurfaceObject_t o_texture, cudaSurfaceObject_t o_depth, float4 *_hdr, float4 *_vertices, float4 *_normals, float4 *_tangents, float4 *_bvhData, float2 *_uvs, float4 *io_colorArr, vCamera _cam, unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time);
void cu_bindTexture(const float4 *_deviceTexture, const unsigned int _w, const unsigned int _h, const vTextureType &_type);
void cu_setHDRDim(const unsigned int &_w, const unsigned int &_h);
void cu_fillFloat4(float4 *d_ptr, float4 _val, unsigned int _size);
void cu_cleanUp();
