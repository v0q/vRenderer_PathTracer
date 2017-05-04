///
/// \file vDataTypes.h
/// \brief Simple data structures to hold the mesh (and other) data on the CPU
/// \authors Teemu Lindborg
/// \version 1.0
/// \date 04/05/17 Updated to NCCA Coding standard
/// Revision History :
/// Initial Version 08/12/16
/// \todo -
///

#pragma once

#include <cmath>
#include <vector>

#include <ngl/Vec3.h>

///
/// \brief vHVert Simple vertex structure, contains the position, normal, tangent and uv-coordinates of a vertex
///
typedef struct vHVert
{
	///
	/// \brief m_vert Position of the vertex
	///
	ngl::Vec3 m_vert;

	///
	/// \brief m_normal Normal of the vertex
	///
	ngl::Vec3 m_normal;

	///
	/// \brief m_tangent  Tangent of the vertex
	///
	ngl::Vec3 m_tangent;

	///
	/// \brief m_u U-texture coordinate
	///
	float m_u;

	///
	/// \brief m_v V-texture coordinate
	///
	float m_v;
} vHVert;

///
/// \brief vHTriangle Simple triangle structure containing the vertex indices of a triangle, used to contain normal and other triangle specific data
///
typedef struct vHTriangle
{
	///
	/// \brief m_indices Vertex indices of the triangle
	///
	unsigned int m_indices[3];
} vHTriangle;
