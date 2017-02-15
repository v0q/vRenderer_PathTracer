#pragma once

#include <cmath>
#include <vector>

typedef struct vFloat3
{
	union
	{
		struct
		{
			float x;
			float y;
			float z;
		};
		float v[3];
	};
	vFloat3() {}
	vFloat3(const vFloat3 &_rhs) : x(_rhs.x), y(_rhs.y), z(_rhs.z) {}
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

	vFloat3 operator-(const vFloat3 &_b) const
	{
		return vFloat3(x - _b.x, y - _b.y, z - _b.z);
	}

	vFloat3 operator*(const float &_b) const
	{
		return vFloat3(x * _b, y * _b, z * _b);
	}

	vFloat3 operator*(const vFloat3 &_b) const
	{
		return vFloat3(x * _b.x, y * _b.y, z * _b.z);
	}

	vFloat3 operator+(const vFloat3 &_b) const
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

typedef struct vInt3
{
	union
	{
		struct
		{
			int x;
			int y;
			int z;
		};
		int v[3];
	};
	vInt3() {}
	vInt3(const vInt3 &_rhs) : x(_rhs.x), y(_rhs.y), z(_rhs.z) {}
	vInt3(const vFloat3 &_rhs) : x(_rhs.x), y(_rhs.y), z(_rhs.z) {}
	vInt3(const unsigned int *_rhs) : x(_rhs[0]), y(_rhs[1]), z(_rhs[2]) {}
	vInt3(const int &_x, const int &_y, const int &_z) : x(_x), y(_y), z(_z) {}

	vInt3& operator=(const vInt3 &_b)
	{
		x = _b.x;
		y = _b.y;
		z = _b.z;

		return *this;
	}

	vInt3& operator=(unsigned int _b[3])
	{
		x = _b[0];
		y = _b[1];
		z = _b[2];

		return *this;
	}

	vInt3& operator-=(const vInt3 &_b)
	{
		x -= _b.x;
		y -= _b.y;
		z -= _b.z;

		return *this;
	}

	vInt3& operator*=(const float &_b)
	{
		x *= _b;
		y *= _b;
		z *= _b;

		return *this;
	}

	vInt3& operator+=(const vInt3 &_b)
	{
		x += _b.x;
		y += _b.y;
		z += _b.z;

		return *this;
	}

	vInt3 operator-(const vInt3 &_b) const
	{
		return vInt3(x - _b.x, y - _b.y, z - _b.z);
	}

	vInt3 operator*(const float &_b) const
	{
		return vInt3(x * _b, y * _b, z * _b);
	}

	vInt3 operator*(const vInt3 &_b) const
	{
		return vFloat3(x * _b.x, y * _b.y, z * _b.z);
	}

	vInt3 operator+(const vInt3 &_b) const
	{
		return vInt3(x + _b.x, y + _b.y, z + _b.z);
	}
} vInt3;

typedef struct vHVert {
	vFloat3 m_vert;
} vHVert;

typedef struct vHTriangle {
	unsigned int m_indices[3];
	vFloat3 m_normal;
} vHTriangle;
