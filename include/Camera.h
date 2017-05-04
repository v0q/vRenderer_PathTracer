///
/// \file Camera.h
/// \brief Simple virtual camera class used in the scene
/// \authors Teemu Lindborg
/// \version 1.0
/// \date 04/05/17 Updated to NCCA Coding standard
/// Revision History :
/// Initial Version 08/12/16
/// \todo Implement the "A Realistic Camera Model for Computer Graphics" described at http://www.cs.virginia.edu/~gfx/courses/2003/ImageSynthesis/papers/Cameras/Realistic%20Camera%20Model.pdf
///

#pragma once

#include <ngl/Mat4.h>
#include <ngl/Vec3.h>

///
/// \brief The Camera class Simple camera class that handles movement and rotations, also keeps track whether the camera is dirty
///													e.g. if the renderer needs to clear its buffers and start tracing from the beginning
///
class Camera
{
public:
	///
	/// \brief Camera Default ctor
	///
	Camera();

	///
	/// \brief ~Camera Default dtor
	///
	~Camera() {}

	///
	/// \brief pitch Add pitch to the camera, triggers the dirty flag
	/// \param _angle Pitch to add
	///
	void pitch(const float &_angle);

	///
	/// \brief yaw Add yaw to the camera, triggers the dirty flag
	/// \param _angle Yaw to add
	///
	void yaw(const float &_angle);

	///
	/// \brief moveForward Move forward/backward, triggers the dirty flag
	/// \param _amt Amount to move
	///
	void moveForward(const float &_amt);

	///
	/// \brief changeFov Change the field of view of the camera, triggers the dirty flag
	/// \param _newFov New field of view to use
	///
	void changeFov(const float &_newFov);

	///
	/// \brief getOrig Get the location of the camera
	/// \return Location of the camera in space
	///
	ngl::Vec3 getOrig() const;

	///
	/// \brief getDir Get the direction of the camera
	/// \return The direction, which is the negative forward vector
	///
	ngl::Vec3 getDir() const;

	///
	/// \brief getUp Get the up vector of the camera
	/// \return Vector perpendicular to the forward and right vectors, facing upwards
	///
  ngl::Vec3 getUp() const;

	///
	/// \brief getRight Get the right vector of the camera
	/// \return Vector perpendicular to the forward and up vectors, facing right
	///
  ngl::Vec3 getRight() const;

	///
	/// \brief getFovScale Get the scale used to calculate ray offsets based on the field of view
	/// \return The field of view scale
	///
  float getFovScale() const;

	///
	/// \brief consume Update all the vectors and values and reset the dirty flag
	///
	void consume();

	///
	/// \brief isDirty Used to check if the camera has updates that need to be consumed
	/// \return Whether the camera is considered dirty
	///
	bool isDirty() const;

private:
	///
	/// \brief setCameraMatrix Update the camera matrix
	///
	void setCameraMatrix();

	///
	/// \brief m_cam Camera matrix
	///
	ngl::Mat4 m_cam;

	///
	/// \brief m_loc Location of the camera
	///
	ngl::Vec3 m_loc;

	///
	/// \brief m_upV Up vector of the camera
	///
	ngl::Vec3 m_upV;

	///
	/// \brief m_rightV Right vector of the camera
	///
	ngl::Vec3 m_rightV;

	///
	/// \brief m_lookAt Initial camera look at, used to calculate the forward vector
	///
	ngl::Vec3 m_lookAt;

	///
	/// \brief m_forwardV Forward vector of the camera
	///
	ngl::Vec3 m_forwardV;

	///
	/// \brief m_fov Used field of view
	///
	float m_fov;

	///
	/// \brief m_pitch Current pitch of the camera
	///
	float m_pitch;

	///
	/// \brief m_yaw Current yaw of the camera
	///
  float m_yaw;

	///
	/// \brief m_isDirty Whether the camera matrix needs to be updated or not
	///
	bool m_isDirty;
};
