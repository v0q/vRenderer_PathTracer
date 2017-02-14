#pragma once

typedef struct vBVHNode {
	// bounding box
	float3 m_bottom;
	float3 m_top;

	// parameters for leafnodes and innernodes occupy same space (union) to save memory
	// top bit discriminates between leafnode and innernode
	// no pointers, but indices (int): faster

	union {
		// inner node - stores indexes to array of CacheFriendlyBVHNode
		struct {
			unsigned int m_leftIndex;
			unsigned int m_rightIndex;
		} m_inner;
		// leaf node: stores triangle count and starting index in triangle list
		struct {
			unsigned int m_count; // Top-most bit set, leafnode if set, innernode otherwise
			unsigned int m_startIndexInTriIndexList;
		} m_leaf;
	} m_u;
} vCFBVHNode;

typedef struct vVert {
	float4 m_vert;
	float4 m_normal;
} vVert;

//typedef struct vTriangle {
//	vVert m_v1;
//	vVert m_v2;
//	vVert m_v3;
//} vTriangle;

typedef struct vTriangle {
	uint3 m_indices;
	float4 m_normal;
	float4 m_center;
	float4 m_bottom;
	float4 m_top;
} vTriangle;

typedef struct vBVH {
	float4 m_normal[7];
	float m_dNear[7];
	float m_dFar[7];
} vBVH;


typedef struct vMesh {
	vTriangle *m_mesh;
	vBVH m_bvh;
	unsigned int m_triCount;
} vMesh;

typedef struct vHitData {
	float4 m_hitPoint;
	float4 m_normal;
	float4 m_emission;
	float4 m_color;
} vHitData;

void cu_runRenderKernel(cudaSurfaceObject_t _texture, float4 *_triangleData, unsigned int *_triIdxList, float2 *_bvhLimits, uint4 *_bvhChildrenOrTriangles,
												unsigned int _triCount, unsigned int _bvhBoxCount, unsigned int _triIdxCount,
												float4 *_colorArr, float4 *_cam, float4 *_dir,
												unsigned int _w, unsigned int _h, unsigned int _frame, unsigned int _time);
void cu_updateBVHBoxCount(unsigned int _bvhBoxes);
void cu_fillFloat4(float4 *d_ptr, float4 _val, unsigned int _size);
