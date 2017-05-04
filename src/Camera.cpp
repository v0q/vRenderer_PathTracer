///
/// \file Camera.cpp
/// \brief Implements the Camera class
///

#include <cmath>
#include "Camera.h"

constexpr float kDegInRad = M_PI/180.f;

Camera::Camera() :
	m_loc(ngl::Vec3(0.f, 0.f, 150.f)),
	m_upV(ngl::Vec3(0.f, 1.f, 0.f)),
	m_lookAt(ngl::Vec3(0.f, 0.f, -1.f)),
  m_fov(75.f),
	m_pitch(0.f),
	m_yaw(0.f),
	m_isDirty(false)
{
	m_forwardV = m_loc - m_lookAt;
  m_forwardV.normalize();

	setCameraMatrix();
}

void Camera::setCameraMatrix()
{
  // To get the right vector, cross an "arbitrary" vector that's different in only y
  // to the forward vector with the forward vector
  m_rightV.cross(m_forwardV + ngl::Vec3(0.f, 1.f, 0.f), m_forwardV);
  m_rightV.normalize();

  // No need to normalize as we're crossing two normalized vectors
  m_upV.cross(m_forwardV, m_rightV);

	m_cam.m_00 = m_rightV.m_x;
	m_cam.m_01 = m_upV.m_x;
	m_cam.m_02 = m_forwardV.m_x;

	m_cam.m_10 = m_rightV.m_y;
	m_cam.m_11 = m_upV.m_y;
	m_cam.m_12 = m_forwardV.m_y;

	m_cam.m_20 = m_rightV.m_z;
	m_cam.m_21 = m_upV.m_z;
	m_cam.m_22 = m_forwardV.m_z;

	m_cam.m_03 = m_loc.m_x;
	m_cam.m_13 = m_loc.m_y;
	m_cam.m_23 = m_loc.m_z;
}

void Camera::pitch(const float &_angle)
{
	// Update the pitch and do some sanity checking/clamping and trigger the dirty flag
  m_pitch += _angle;
	m_pitch = std::fabs(m_pitch) > M_PI_2 ? (m_pitch < 0 ? -M_PI_2 : M_PI_2) : m_pitch;
	m_isDirty = true;
}

void Camera::yaw(const float &_angle)
{
	m_yaw += _angle;
	m_isDirty = true;
}

void Camera::moveForward(const float &_x)
{
	m_loc += m_forwardV * _x;
	m_isDirty = true;
}

void Camera::changeFov(const float &_newFov)
{
	m_fov = _newFov;
	m_isDirty = true;
}

void Camera::consume()
{
	// Calculate the new forward vector from the yaw and pitch
	float sy = std::sin(m_yaw);
	float cy = std::cos(m_yaw);
  float sp = std::sin(m_pitch);
	float cp = std::cos(m_pitch);

	m_forwardV = ngl::Vec3(sy*cp, sp, cy*cp);
  m_forwardV.normalize();
	setCameraMatrix();

	m_isDirty = false;
}

bool Camera::isDirty() const
{
	return m_isDirty;
}

ngl::Vec3 Camera::getOrig() const
{
  return m_loc;
}

ngl::Vec3 Camera::getDir() const
{
  return -m_forwardV;
}

ngl::Vec3 Camera::getUp() const
{
  return m_upV;
}

ngl::Vec3 Camera::getRight() const
{
  return m_rightV;
}

float Camera::getFovScale() const
{
  float fovRad = m_fov * kDegInRad;
	return std::tan(fovRad/2.f);
}
