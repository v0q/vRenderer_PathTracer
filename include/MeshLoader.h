#pragma once

#include <GL/glew.h>
#include <assimp/scene.h>
#include <string>

#ifdef __VRENDERER_OPENCL__
	#ifdef __APPLE__
		#include <OpenCL/cl.h>
	#else
		#include <CL/cl.h>
	#endif
	typedef cl_float4 ftype;
#elif __VRENDERER_CUDA__
	#include <cuda/cuda_runtime.h>
	typedef float4 ftype;
#endif

typedef struct vFloat3
{
	float x;
	float y;
	float z;
	vFloat3(const float &_x, const float &_y, const float &_z) : x(_x), y(_y), z(_z) {}
} vFloat3;

///
/// \brief The vBoundingBox class Simple class for calculating the bounding box of a mesh
///
class vBB
{
public:
	///
	/// @brief vBoundingBox Default ctor
	///
	vBB() :
		m_initialised(false)
	{
		m_x.m_min = m_x.m_max = 0.f;
		m_y.m_min = m_y.m_max = 0.f;
		m_z.m_min = m_z.m_max = 0.f;
	}

	///
	/// @brief extendBB Simple function to update the extents of the bounding box if needed
	/// @param _vert Mesh vertex to evaluate against the current bounding box
	///
	void extendBB(aiVector3t<float> &_vert) {
		if(m_initialised)
		{
			m_x.m_min = m_x.m_min < _vert.x ? m_x.m_min : _vert.x;
			m_x.m_max = m_x.m_max > _vert.x ? m_x.m_max : _vert.x;

			m_y.m_min = m_y.m_min < _vert.y ? m_y.m_min : _vert.y;
			m_y.m_max = m_y.m_max > _vert.y ? m_y.m_max : _vert.y;

			m_z.m_min = m_z.m_min < _vert.z ? m_z.m_min : _vert.z;
			m_z.m_max = m_z.m_max > _vert.z ? m_z.m_max : _vert.z;
		}
		else
		{
			m_x.m_min = m_x.m_max = _vert.x;
			m_y.m_min = m_y.m_max = _vert.y;
			m_z.m_min = m_z.m_max = _vert.z;

			m_initialised = true;
		}
	}

	vFloat3 getMinBounds() const {
		return vFloat3(m_x.m_min, m_y.m_min, m_z.m_min);
	}

	vFloat3 getMaxBounds() const {
		return vFloat3(m_x.m_max, m_y.m_max, m_z.m_max);
	}

	void print() const
	{
		std::cout << "X: [" << m_x.m_min << ", " << m_x.m_max << "]\n";
		std::cout << "Y: [" << m_y.m_min << ", " << m_y.m_max << "]\n";
		std::cout << "Z: [" << m_z.m_min << ", " << m_z.m_max << "]\n";
	}

private:
	bool m_initialised;
	union
	{
		struct
		{
			float m_min;
			float m_max;
		} m_x, m_y, m_z;
	};
};

class vMeshLoader
{
public:
	vMeshLoader(const std::string &_mesh);
	~vMeshLoader();
	static std::vector<vFloat3> loadMesh(const std::string &_mesh);
private:
};
