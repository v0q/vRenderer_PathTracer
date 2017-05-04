///
/// \file NGLScene.h
/// \brief This class inherits from the Qt OpenGLWindow and allows us to use NGL to draw OpenGL
///				 Creates the renderer, handles data propagation from the UI to the renderer and
///				 creates OpenGL textures and VBO's to visualise the renderer output
/// \authors Jonathan Macey, Teemu Lindborg
/// \version 1.0
/// \date 10/9/13
/// Revision History :
/// This is an initial version used for the new NGL6 / Qt 5 demos
/// Initial Version 08/12/16
/// Updated to NCCA Coding standard 04/05/17
/// \todo General cleanup and move the pass through slots from the scene to the renderer
///				Switch to OpenImageIO for HDRI and texture loading. Bring in colour management options with OpenColorIO
///

#pragma once

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
/// \class NGLScene
/// \brief our main glwindow widget for NGL applications all drawing elements are
/// put in this file
//----------------------------------------------------------------------------------------------------------------------

class NGLScene : public QOpenGLWidget
{
  Q_OBJECT
  public:
    //----------------------------------------------------------------------------------------------------------------------
		/// \brief ctor for our NGL drawing class
		/// \param [in] parent the parent window to the class
    //----------------------------------------------------------------------------------------------------------------------
    NGLScene(QWidget *_parent = nullptr);
    //----------------------------------------------------------------------------------------------------------------------
		/// \brief dtor must close down ngl and release OpenGL resources
    //----------------------------------------------------------------------------------------------------------------------
    ~NGLScene();
    //----------------------------------------------------------------------------------------------------------------------
		/// \brief the initialize class is called once when the window is created and we have a valid GL context
    /// use this to setup any default GL stuff
    //----------------------------------------------------------------------------------------------------------------------
    void initializeGL();
    //----------------------------------------------------------------------------------------------------------------------
		/// \brief this is called everytime we want to draw the scene
    //----------------------------------------------------------------------------------------------------------------------
    void paintGL();

		void changeRenderChannel() { m_renderChannel ^= 1; }

private:
    //----------------------------------------------------------------------------------------------------------------------
		/// \brief this is called everytime we resize the window
    //----------------------------------------------------------------------------------------------------------------------
    void resizeGL(int _w, int _h);

    //----------------------------------------------------------------------------------------------------------------------
		/// \brief this method is called every time a mouse is moved
		/// \param _event the Qt Event structure
    //----------------------------------------------------------------------------------------------------------------------
    void mouseMoveEvent (QMouseEvent * _event );
    //----------------------------------------------------------------------------------------------------------------------
		/// \brief this method is called everytime the mouse button is pressed
    /// inherited from QObject and overridden here.
		/// \param _event the Qt Event structure
    //----------------------------------------------------------------------------------------------------------------------
    void mousePressEvent ( QMouseEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
		/// \brief this method is called everytime the mouse button is released
    /// inherited from QObject and overridden here.
		/// \param _event the Qt Event structure
    //----------------------------------------------------------------------------------------------------------------------
		void mouseReleaseEvent ( QMouseEvent *_event );

    //----------------------------------------------------------------------------------------------------------------------
		/// \brief this method is called everytime the mouse wheel is moved
    /// inherited from QObject and overridden here.
		/// \param _event the Qt Event structure
    //----------------------------------------------------------------------------------------------------------------------
    void wheelEvent( QWheelEvent *_event);
		void timerEvent(QTimerEvent *_event);
		/// \brief windows parameters for mouse control etc.
    WinParams m_win;
    /// position for our model
    ngl::Vec3 m_modelPos;

		///
		/// \brief loadTexture Prompts the texture selection and tries to load it using QImage, passes the QImage to the renderer if loaded successfully
		/// \param _type Type of the texture, 0 = Diffuse, 1 = Normal, 2 = Specular
		///
		void loadTexture(const unsigned int &_type);

		///
		/// \brief m_vao Vertex array object for a screen quad
		///
		GLuint m_vao;

		///
		/// \brief m_vbo Vertex buffer object for a screen quad
		///
		GLuint m_vbo;

		///
		/// \brief m_texture Texture component that the renderer output will be drawn to
		///
		GLuint m_texture;

		///
		/// \brief m_depthTexture Texture depth component, renderer draws the depth to this texture. Not used for anything at the moment
		///
		GLuint m_depthTexture;

		///
		/// \brief m_renderer Abstract renderer, the type of the renderer is selected using qmake options
		///
    std::unique_ptr<vRenderer> m_renderer;

		///
		/// \brief m_virtualCamera Virtual camera used by the renderer
		///
		Camera *m_virtualCamera;

		///
		/// \brief m_fxaaEnabled Should the final output be drawn with FXAA
		///
		int m_fxaaEnabled;

		///
		/// \brief m_renderChannel Whether to draw the final output or the depth
		///
		int m_renderChannel;

		///
		/// \brief m_fxaaSharpness FXAA sharpness from the UI, passed to the shader when drawing
		///
		float m_fxaaSharpness;

		///
		/// \brief m_fxaaSubpixQuality FXAA subpixel quality from the UI, passed to the shader when drawing
		///
		float m_fxaaSubpixQuality;

		///
		/// \brief m_fxaaEdgeThreshold FXAA edge threshold from the UI, passed to the shader when drawing
		///
		float m_fxaaEdgeThreshold;

		//----------------------------------------------------------------------------------------------------------------------
		/// \brief class for text rendering
		//----------------------------------------------------------------------------------------------------------------------
		std::unique_ptr<ngl::Text> m_text;
		//----------------------------------------------------------------------------------------------------------------------
		/// \brief flag for the fps timer
		//----------------------------------------------------------------------------------------------------------------------
		int m_fpsTimer;
		//----------------------------------------------------------------------------------------------------------------------
		/// \brief the fps to draw
		//----------------------------------------------------------------------------------------------------------------------
		int m_fps;
		//----------------------------------------------------------------------------------------------------------------------
		/// \brief number of frames for the fps counter
		//----------------------------------------------------------------------------------------------------------------------
		int m_frames;
		//----------------------------------------------------------------------------------------------------------------------
		/// \brief timer for re-draw
		//----------------------------------------------------------------------------------------------------------------------
		QTime m_timer;

public slots:
		///
		/// \brief loadMesh Loads a mesh, generates a SBVH acceleration structure for it and passes the data the renderer
		///
		void loadMesh();

		///
		/// \brief loadHDR Loads an EXR and passes the data to the renderer
		///
		void loadHDR();

		///
		/// \brief loadBRDF Loads MERL BRDF data and passes it to the renderer
		///
		void loadBRDF();

		///
		/// \brief useBRDF Passes an UI signal to the renderer whether to use the BRDF data or not
		/// \param _val Whether to use BRDF or not
		///
		void useBRDF(const bool &_val);

		///
		/// \brief useExampleSphere Passes an UI signal to the renderer whether to use an example sphere
		/// \param _val Whether to use the example shere or not
		///
		void useExampleSphere(const bool &_val);

		///
		/// \brief useCornellEnv Passes an UI signal to the renderer whether to use the Cornell box or an HDRI environment
		/// \param _val
		///
		void useCornellEnv(const bool &_val);

		///
		/// \brief loadDiffuse Pass through function to load in a diffuse texture
		///
		void loadDiffuse() { loadTexture(0); }

		///
		/// \brief loadNormal Pass through function to load in a normal texture
		///
		void loadNormal() { loadTexture(1); }

		///
		/// \brief loadSpecular Pass through function to load in a specular texture
		///
		void loadSpecular() { loadTexture(2); }

		///
		/// \brief changeFov Updates the FOV from the UI to the renderer
		/// \param _newFov New field of view to be used
		///
		void changeFov(const int &_newFov);

		///
		/// \brief changeFresnelCoef Updates the fresnel coef from the UI to the renderer
		/// \param _newVal New fresnel coef
		///
		void changeFresnelCoef(const int &_newVal);

		///
		/// \brief changeFresnelPower Updates the fresnel power from the UI to the renderer
		/// \param _newVal New fresnel power
		///
		void changeFresnelPower(const int &_newVal);

		///
		/// \brief toggleFXAA Enable/disable FXAA based on the UI input
		/// \param _enabled Whether to use FXAA or not
		///
		void toggleFXAA(const bool &_enabled) { m_fxaaEnabled = _enabled ? 1 : 0; }

		///
		/// \brief fxaaSharpness Updates the FXAA sharpness shader uniform
		/// \param _newVal New FXAA sharpness value
		///
		void fxaaSharpness(const int &_newVal) { m_fxaaSharpness = _newVal/100.f; }

		///
		/// \brief fxaaSubpixQuality Updates the FXAA subpixel quality shader uniform
		/// \param _newVal New FXAA subpixel quality value
		///
		void fxaaSubpixQuality(const int &_newVal) { m_fxaaSubpixQuality = _newVal/100.f; }

		///
		/// \brief fxaaEdgeThreshold Updates the FXAA edge threshold shader uniform
		/// \param _newVal New FXAA edge threshold value
		///
		void fxaaEdgeThreshold(const int &_newVal) { m_fxaaEdgeThreshold = _newVal/1000.f; }

signals:
		///
		/// \brief meshLoaded Signals the UI that a mesh has been loaded successfully
		///
    void meshLoaded(const QString &);

		///
		/// \brief textureLoaded Signals the UI that a texture has been loaded successfully
		///
		void textureLoaded(const QString &, const unsigned int &);

		///
		/// \brief brdfLoaded Signals the UI that MERL BRDF data has been loaded successfully
		///
		void brdfLoaded(const QString &);

		///
		/// \brief HDRILoaded Signals the UI that a HDRI map has been loaded successfully
		///
		void HDRILoaded(const QString &);
};
