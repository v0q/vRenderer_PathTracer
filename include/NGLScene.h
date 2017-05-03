#ifndef NGLSCENE_H_
#define NGLSCENE_H_

#include <iostream>
#include <memory>
#include <ngl/Vec3.h>
#include <ngl/Text.h>

#include <QOpenGLWidget>
#include <QTime>

#include "Camera.h"
#include "WindowParams.h"

class vRenderer;

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

class NGLScene : public QOpenGLWidget
{
  Q_OBJECT
  public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief ctor for our NGL drawing class
    /// @param [in] parent the parent window to the class
    //----------------------------------------------------------------------------------------------------------------------
    NGLScene(QWidget *_parent = nullptr);
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

		void changeRenderChannel() { m_renderChannel ^= 1; }

private:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this is called everytime we resize the window
    //----------------------------------------------------------------------------------------------------------------------
    void resizeGL(int _w, int _h);

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

		void loadTexture(const unsigned int &_type);
    /// VerterArrayObject and VertexBufferObject for our screen quad
		GLuint m_vao, m_vbo, m_texture, m_depthTexture;
		bool m_renderTexture;

    std::unique_ptr<vRenderer> m_renderer;
		Camera *m_virtualCamera;

		int m_fxaaEnabled;
		int m_renderChannel;
		float m_yaw;
		float m_pitch;

		float m_fxaaSharpness;
		float m_fxaaSubpixQuality;
		float m_fxaaEdgeThreshold;

		float *m_brdf;

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

public slots:
		void loadMesh();
		void loadHDR();
		void loadBRDF();

		void useBRDF(const bool &_val);
		void useExampleSphere(const bool &_val);
		void useCornellEnv(const bool &_val);

		void loadDiffuse() { loadTexture(0); }
		void loadNormal() { loadTexture(1); }
		void loadSpecular() { loadTexture(2); }

		void changeFov(const int &_newFov);
		void changeFresnelCoef(const int &_newVal);
		void changeFresnelPower(const int &_newVal);

		void toggleFXAA(const bool &_enabled) { m_fxaaEnabled = _enabled ? 1 : 0; std::cout << m_fxaaEnabled << "\n"; }
		void fxaaSharpness(const int &_newVal) { m_fxaaSharpness = _newVal/100.f; std::cout << m_fxaaSharpness << "\n"; }
		void fxaaSubpixQuality(const int &_newVal) { m_fxaaSubpixQuality = _newVal/100.f; std::cout << m_fxaaSubpixQuality << "\n"; }
		void fxaaEdgeThreshold(const int &_newVal) { m_fxaaEdgeThreshold = _newVal/1000.f; std::cout << m_fxaaEdgeThreshold << "\n"; }

signals:
    void meshLoaded(const QString &);
		void textureLoaded(const QString &, const unsigned int &);
		void brdfLoaded(const QString &);
		void HDRILoaded(const QString &);
};

#endif
