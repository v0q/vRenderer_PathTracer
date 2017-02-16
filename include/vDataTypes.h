#pragma once

#include <cmath>
#include <vector>

#include <ngl/Vec3.h>

typedef struct vHVert {
	ngl::Vec3 m_vert;
} vHVert;

typedef struct vHTriangle {
	unsigned int m_indices[3];
	ngl::Vec3 m_normal;
} vHTriangle;
