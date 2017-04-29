#pragma once

#include <cmath>
#include <vector>

#include <ngl/Vec3.h>

typedef struct vHVert {
	ngl::Vec3 m_vert;
	ngl::Vec3 m_normal;
	ngl::Vec3 m_tangent;
	// Texture coordinates
	float m_u;
	float m_v;
} vHVert;

typedef struct vHTriangle {
	unsigned int m_indices[3];
} vHTriangle;
