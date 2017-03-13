#pragma once

#include <ngl/Mat4.h>
#include <ngl/Vec3.h>

class Camera
{
public:
	Camera();
	~Camera() {}

	void pitch(const float &_angle);
	void yaw(const float &_angle);
	void moveForward(const float &_amt);

	ngl::Vec3 getOrig() const;
	ngl::Vec3 getDir() const;
  ngl::Vec3 getUp() const;
  ngl::Vec3 getRight() const;
  float getFovScale() const;

	void consume();
	bool isDirty() const;

private:
	void setCameraMatrix();

	ngl::Mat4 m_cam;

	ngl::Vec3 m_loc;
	ngl::Vec3 m_upV;
	ngl::Vec3 m_rightV;
	ngl::Vec3 m_lookAt;
	ngl::Vec3 m_forwardV;

	float m_fov;
	float m_pitch;
  float m_yaw;
	bool m_isDirty;
};
