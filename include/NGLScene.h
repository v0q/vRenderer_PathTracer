#ifndef NGLSCENE_H_
#define NGLSCENE_H_
#include <iostream>
#include <memory>
#include <ngl/Vec3.h>
#include <ngl/Text.h>

#include <QOpenGLWindow>
#include <QTime>
#include <cuda_gl_interop.h>
#include "WindowParams.h"
//----------------------------------------------------------------------------------------------------------------------
/// @file NGLScene.h
/// @brief this class inherits from the Qt OpenGLWindow and allows us to use NGL to draw OpenGL
/// @author Jonathan Macey
/// @version 1.0
/// @date 10/9/13
/// Revision History :
/// This is an initial version used for the new NGL6 / Qt 5 demos
/// @class NGLScene
/// @brief our main glwindow widget for NGL applications all drawing elements are
/// put in this file
//----------------------------------------------------------------------------------------------------------------------

class NGLScene : public QOpenGLWindow
{
  public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief ctor for our NGL drawing class
    /// @param [in] parent the parent window to the class
    //----------------------------------------------------------------------------------------------------------------------
    NGLScene();
    //----------------------------------------------------------------------------------------------------------------------
		/// @brief dtor must close down ngl and release OpenGL resources
    //----------------------------------------------------------------------------------------------------------------------
    ~NGLScene();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the initialize class is called once when the window is created and we have a valid GL context
    /// use this to setup any default GL stuff
    //----------------------------------------------------------------------------------------------------------------------
    void initializeGL();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this is called everytime we want to draw the scene
    //----------------------------------------------------------------------------------------------------------------------
    void paintGL();

private:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this is called everytime we resize the window
    //----------------------------------------------------------------------------------------------------------------------
    void resizeGL(int _w, int _h);

    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Qt Event called when a key is pressed
    /// @param [in] _event the Qt event to query for size etc
    //----------------------------------------------------------------------------------------------------------------------
    void keyPressEvent(QKeyEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this method is called every time a mouse is moved
    /// @param _event the Qt Event structure
    //----------------------------------------------------------------------------------------------------------------------
    void mouseMoveEvent (QMouseEvent * _event );
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this method is called everytime the mouse button is pressed
    /// inherited from QObject and overridden here.
    /// @param _event the Qt Event structure
    //----------------------------------------------------------------------------------------------------------------------
    void mousePressEvent ( QMouseEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this method is called everytime the mouse button is released
    /// inherited from QObject and overridden here.
    /// @param _event the Qt Event structure
    //----------------------------------------------------------------------------------------------------------------------
    void mouseReleaseEvent ( QMouseEvent *_event );

    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this method is called everytime the mouse wheel is moved
    /// inherited from QObject and overridden here.
    /// @param _event the Qt Event structure
    //----------------------------------------------------------------------------------------------------------------------
    void wheelEvent( QWheelEvent *_event);
		void timerEvent(QTimerEvent *_event);
    /// @brief windows parameters for mouse control etc.
    WinParams m_win;
    /// position for our model
    ngl::Vec3 m_modelPos;

		void validateCuda(cudaError_t _err) {
			if(_err != cudaSuccess) {
				std::cerr << "Failed to perform a cuda operation\n";
				exit(0);
			}
		}

		/// VerterArrayObject and VertexBufferObject for our screen quad
		GLuint m_vao, m_vbo, m_texture;
		cudaGraphicsResource_t m_cudaGLTextureBuffer;
		cudaArray *m_cudaImgArray;
		float3 *m_camera;
		float3 *m_camdir;
		float3 *m_colorArray;
		unsigned int m_frame;

		//----------------------------------------------------------------------------------------------------------------------
		/// @brief class for text rendering
		//----------------------------------------------------------------------------------------------------------------------
		std::unique_ptr<ngl::Text> m_text;
		//----------------------------------------------------------------------------------------------------------------------
		/// @brief flag for the fps timer
		//----------------------------------------------------------------------------------------------------------------------
		int m_fpsTimer;
		//----------------------------------------------------------------------------------------------------------------------
		/// @brief the fps to draw
		//----------------------------------------------------------------------------------------------------------------------
		int m_fps;
		//----------------------------------------------------------------------------------------------------------------------
		/// @brief number of frames for the fps counter
		//----------------------------------------------------------------------------------------------------------------------
		int m_frames;
		//----------------------------------------------------------------------------------------------------------------------
		/// @brief timer for re-draw
		//----------------------------------------------------------------------------------------------------------------------
		QTime m_timer;
};



#endif
