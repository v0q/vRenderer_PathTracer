#pragma once

#include <cmath>
#include <vector>

typedef struct vFloat3
{
	float x;
	float y;
	float z;
	vFloat3() {}
	vFloat3(const float &_x, const float &_y, const float &_z) : x(_x), y(_y), z(_z) {}

	vFloat3& operator=(const vFloat3 &_b)
	{
		x = _b.x;
		y = _b.y;
		z = _b.z;

		return *this;
	}

	vFloat3& operator-=(const vFloat3 &_b)
	{
		x -= _b.x;
		y -= _b.y;
		z -= _b.z;

		return *this;
	}

	vFloat3& operator*=(const float &_b)
	{
		x *= _b;
		y *= _b;
		z *= _b;

		return *this;
	}

	vFloat3& operator+=(const vFloat3 &_b)
	{
		x += _b.x;
		y += _b.y;
		z += _b.z;

		return *this;
	}

	vFloat3 operator-(const vFloat3 &_b)
	{
		return vFloat3(x - _b.x, y - _b.y, z - _b.z);
	}

	vFloat3 operator*(const float &_b)
	{
		return vFloat3(x * _b, y * _b, z * _b);
	}

	vFloat3 operator+(const vFloat3 &_b)
	{
		return vFloat3(x + _b.x, y + _b.y, z + _b.z);
	}

	float length()
	{
		return std::sqrt(x*x + y*y + z*z);
	}

	void normalize()
	{
		float len = length();
		if(len)
		{
			x /= len;
			y /= len;
			z /= len;
		}
	}
} vFloat3;

typedef struct vHVert {
	vFloat3 m_vert;
	vFloat3 m_normal;
} vHVert;

typedef struct vHTriangle {
	unsigned int m_indices[3];
	vFloat3 m_normal;
	vFloat3 m_center;

	// Raytracing intersection pre-computed cache:
	float m_d;
	float m_d1;
	float m_d2;
	float m_d3;
	vFloat3 m_e1;
	vFloat3 m_e2;
	vFloat3 m_e3;

	// bounding box
	vFloat3 m_bottom;
	vFloat3 m_top;
} vHTriangle;
