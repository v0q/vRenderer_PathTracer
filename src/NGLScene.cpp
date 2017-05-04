///
/// \file NGLScene.cpp
/// \brief Implements the NGLScene, OpenGL and Cuda/OpenCL interop etc
///

#include <QMouseEvent>
#include <QFileDialog>
#include <QImageReader>
#include <ngl/ShaderLib.h>
#include <iostream>
#include <vector>
#include <assert.h>
#include <chrono>
#include <memory>
#include <fstream>
#include <OpenEXR/ImfRgba.h>
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImathBox.h>

//#include <OpenImageIO/imageio.h>
//#include <OpenColorIO/OpenColorIO.h>

#include "NGLScene.h"
#include <ngl/NGLInit.h>

#include "MeshLoader.h"
#include "BRDFLoader.h"

#ifdef __VRENDERER_CUDA__
	#include "vRendererCuda.h"
#elif __VRENDERER_OPENCL__
	#include "vRendererCL.h"
#endif

//namespace OCIO = OCIO_NAMESPACE;

NGLScene::NGLScene(QWidget *_parent) :
  QOpenGLWidget(_parent),
	m_modelPos(ngl::Vec3(0.0f, 0.0f, 0.0f)),
  m_fxaaEnabled(0),
	m_renderChannel(0),
	m_fxaaSharpness(0.5f),
	m_fxaaSubpixQuality(0.75f),
	m_fxaaEdgeThreshold(0.166f)
{
	m_fpsTimer = startTimer(0);
	m_fps = 0;
	m_frames = 0;
	m_timer.start();

	m_virtualCamera = new Camera;
}

NGLScene::~NGLScene()
{
	// Cleanup, renderer is deleted automatically because of smart pointers
	delete m_virtualCamera;
  glDeleteBuffers(1, &m_vbo);
	glDeleteVertexArrays(1, &m_vao);
	glDeleteTextures(1, &m_texture);
	glDeleteTextures(1, &m_depthTexture);

	std::cout << "Shutting down NGL, removing VAO's and Shaders\n";
}

void NGLScene::resizeGL(int _w , int _h)
{
  m_win.width  = static_cast<int>( _w * devicePixelRatio() );
  m_win.height = static_cast<int>( _h * devicePixelRatio() );
}


void NGLScene::initializeGL()
{
  // we need to initialise the NGL lib which will load all of the OpenGL functions, this must
  // be done once we have a valid GL context but before we call any GL commands. If we dont do
  // this everything will crash
  ngl::NGLInit::instance();

	// Create/initialise the renderer based on which one's selected
#ifdef __VRENDERER_CUDA__
  m_renderer.reset(new vRendererCuda);
#elif __VRENDERER_OPENCL__
  m_renderer.reset(new vRendererCL);
#endif

	// Initialise the renderer and connect our virtual camera to it
	m_renderer->init((unsigned int)width(), (unsigned int)height());
	m_renderer->setCamera(m_virtualCamera);

	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

	// No need to do depth testing or multisampling as we're only drawing a texture to a screen quad
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_MULTISAMPLE);

	ngl::ShaderLib *shader = ngl::ShaderLib::instance();

	/// Load in the normal screen quad and FXAA shaders
	// Regular screen quad
	{
		shader->createShaderProgram("Screen Quad");

		shader->attachShader("VertexShader", ngl::ShaderType::VERTEX);
		shader->attachShader("FragmentShader", ngl::ShaderType::FRAGMENT);

		shader->loadShaderSource("VertexShader", "shaders/screenQuad.vert");
		shader->loadShaderSource("FragmentShader", "shaders/screenQuad.frag");

		shader->compileShader("VertexShader");
		shader->compileShader("FragmentShader");

		shader->attachShaderToProgram("Screen Quad", "VertexShader");
		shader->attachShaderToProgram("Screen Quad", "FragmentShader");

		shader->linkProgramObject("Screen Quad");
	}

	// FXAA
	{
		shader->createShaderProgram("FXAA");

		shader->attachShader("VertexShaderFXAA", ngl::ShaderType::VERTEX);
		shader->attachShader("FragmentShaderFXAA", ngl::ShaderType::FRAGMENT);

		shader->loadShaderSource("VertexShaderFXAA", "shaders/screenQuad.vert");
		shader->loadShaderSource("FragmentShaderFXAA", "shaders/screenQuadFXAA.frag");

		shader->compileShader("VertexShaderFXAA");
		shader->compileShader("FragmentShaderFXAA");

		shader->attachShaderToProgram("FXAA", "VertexShaderFXAA");
		shader->attachShaderToProgram("FXAA", "FragmentShaderFXAA");

		shader->linkProgramObject("FXAA");
	}

	shader->use("Screen Quad");

	// Screen quad data
	float vertices[] = {
		// First triangle
		-1.0f,  1.0f,
		-1.0f, -1.0f,
		 1.0f,  1.0f,
		// Second triangle
		-1.0f, -1.0f,
		 1.0f, -1.0f,
		 1.0f,  1.0f
	};

	float uvs[] = {
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f
	};

	// Generate the VAO, VBO and textures
	glGenVertexArrays(1, &m_vao);
	glBindVertexArray(m_vao);

	glGenBuffers(1, &m_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices)+sizeof(uvs), 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), &vertices);
	glBufferSubData(GL_ARRAY_BUFFER, sizeof(vertices), sizeof(uvs), &uvs);

	glGenTextures(1, &m_texture);
	glBindTexture(GL_TEXTURE_2D, m_texture);

	// set basic parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Create texture data (4-component unsigned byte)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width(), height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glGenTextures(1, &m_depthTexture);
	glBindTexture(GL_TEXTURE_2D, m_depthTexture);

	// set basic parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Create texture data (4-component unsigned byte)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width(), height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	// Register the texture and depth texture with the renderer
  m_renderer->registerTextureBuffer(m_texture);
	m_renderer->registerDepthBuffer(m_depthTexture);

  // Unbind the texture
	glBindTexture(GL_TEXTURE_2D, 0);

	m_text.reset(new ngl::Text(QFont("Arial", 12)));
	m_text->setScreenSize(width(), height());

	// Load in an initial HDRI map using OpenEXR
	Imf::Rgba *pixelBuffer;
	try
	{
		Imf::RgbaInputFile in("hdr/Arches_E_PineTree_3k.exr");
		Imath::Box2i win = in.dataWindow();

		Imath::V2i dim(win.max.x - win.min.x + 1,
									 win.max.y - win.min.y + 1);

		pixelBuffer = new Imf::Rgba[dim.x *dim.y];

		int dx = win.min.x;
		int dy = win.min.y;

		in.setFrameBuffer(pixelBuffer - dx - dy * dim.x, 1, dim.x);
		in.readPixels(win.min.y, win.max.y);

		// Send the HDRI data to the renderer and signal the UI
		m_renderer->loadHDR(pixelBuffer, dim.x, dim.y);
		emit(HDRILoaded("hdr/Arches_E_PineTree_3k.exr"));
	}
	catch (Iex::BaseExc &e)
	{
		std::cerr << e.what() << "\n";
		exit(0);
	}
}

void NGLScene::timerEvent(QTimerEvent *_event)
{
	// Update the FPS timer and call update()
	if(_event->timerId() == m_fpsTimer)
	{
		if(m_timer.elapsed() > 1000.0)
		{
			m_fps = m_frames;
			m_frames = 0;
			m_timer.restart();
		}
	}
	update();
}

void NGLScene::paintGL()
{
	// Clear the screen and depth buffer
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0,0,m_win.width,m_win.height);

	// Time the path trace
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	// Perform a path trace step
	m_renderer->render();

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	// Get an instance to the shaderlib
	ngl::ShaderLib *shader = ngl::ShaderLib::instance();

	// Whether to use FXAA or not
	switch(m_fxaaEnabled)
	{
		case 0:
		default:
			shader->use("Screen Quad");
		break;
		case 1:
			shader->use("FXAA");
		break;
	}

	// Bind the VAO and VBO and draw the screen quad with the output of from the renderer
	glBindVertexArray(m_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

	GLuint posLocation = shader->getAttribLocation("Screen Quad", "a_Position");
	GLuint uvLocation = shader->getAttribLocation("Screen Quad", "a_FragCoord");

	glEnableVertexAttribArray(posLocation);
	glEnableVertexAttribArray(uvLocation);

	glVertexAttribPointer(posLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribPointer(uvLocation, 2, GL_FLOAT, GL_FALSE, 0, (GLvoid *)(12*sizeof(float)));

	// Bind the textures to the shader
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_texture);
	shader->setRegisteredUniform1i("u_ptResult", 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, m_depthTexture);
	shader->setRegisteredUniform1i("u_ptDepth", 1);
	shader->setRegisteredUniform1i("u_channel", m_renderChannel);

	shader->setRegisteredUniform2f("u_screenDim", width(), height());
	shader->setRegisteredUniform2f("u_invScreenDim", 1.f/width(), 1.f/height());

	// Set FXAA uniforms
	if(m_fxaaEnabled)
	{
		shader->setRegisteredUniform1f("u_SubPixQuality", m_fxaaSubpixQuality);
		shader->setRegisteredUniform1f("u_EdgeThreshold", m_fxaaEdgeThreshold);
		shader->setRegisteredUniform1f("u_Sharpness", m_fxaaSharpness);
	}

	// Draw the screen quad
	glDrawArrays(GL_TRIANGLES, 0, 6);

	// Clean up
	shader->useNullProgram();
	glDisableVertexAttribArray(posLocation);
	glDisableVertexAttribArray(uvLocation);

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
	// Render "shadow" first for better visibility against bright backgrounds
	m_text->setColour(0, 0, 0);
	QString text = QString("%1 fps").arg(m_fps);
	m_text->renderText(11, 21, text);
	text = QString("Render time/frame: %1ms").arg(duration);
	m_text->renderText(11, 41, text);
	text = QString("%1 samples per pixel").arg(m_renderer->getFrameCount() * 2);
	m_text->renderText(11, 61, text);

	// Then render the normal text on top of it
	m_text->setColour(1, 1, 1);
	text = QString("%1 fps").arg(m_fps);
	m_text->renderText(10, 20, text);
	text = QString("Render time/frame: %1ms").arg(duration);
	m_text->renderText(10, 40, text);
	text = QString("%1 samples per pixel").arg(m_renderer->getFrameCount() * 2);
	m_text->renderText(10, 60, text);
	++m_frames;
}

void NGLScene::loadMesh()
{
	// Open a file prompt and try to load the mesh
	QString location = QFileDialog::getOpenFileName(this, tr("Load mesh"), NULL, tr("3d models (*.obj *.ply *.fbx)"));
	if(!location.isEmpty())
	{
		vMeshData mesh = vMeshLoader::loadMesh(location.toStdString());

		// For now the implementation has limitation where the tree traversal/GPU data building is not handled correctly if there's only one node
		if(mesh.m_bvh.getNodeCount() > 1)
		{
			m_renderer->initMesh(mesh);
			m_renderer->clearBuffer();

			// Clean up
			emit meshLoaded(mesh.m_name);
		}

		// Clean up the data from the CPU
		mesh.m_bvh.getRoot()->cleanUp();
	}
}

void NGLScene::loadHDR()
{
	// Open a file prompt and try to load an EXR
	QString location = QFileDialog::getOpenFileName(this, tr("Load texture"), NULL, tr("EXR-files (*.exr)"));
	if(!location.isEmpty())
	{
		// Load the EXR and send the data to the renderer
		Imf::Rgba *pixelBuffer;
		try
		{
			Imf::RgbaInputFile in(location.toStdString().c_str());
			Imath::Box2i win = in.dataWindow();

			Imath::V2i dim(win.max.x - win.min.x + 1,
										 win.max.y - win.min.y + 1);

			pixelBuffer = new Imf::Rgba[dim.x *dim.y];

			int dx = win.min.x;
			int dy = win.min.y;

			in.setFrameBuffer(pixelBuffer - dx - dy * dim.x, 1, dim.x);
			in.readPixels(win.min.y, win.max.y);

			m_renderer->loadHDR(pixelBuffer, dim.x, dim.y);
			m_renderer->clearBuffer();
			emit(HDRILoaded(location));
		}
		catch (Iex::BaseExc &e)
		{
			std::cerr << e.what() << "\n";
		}
	}
}

void NGLScene::loadTexture(const unsigned int &_type)
{
	// Open a file prompt and try to load a texture
	QString location = QFileDialog::getOpenFileName(this, tr("Load texture"), NULL, tr("Image files (*.jpg *.jpeg *.tif *.tiff *.png)"));
	if(!location.isEmpty())
	{
		// Read the image using QImage and send the data to the renderer
		QImageReader reader(location);
		QImage texture(location);
		if(texture.isNull())
		{
			std::cerr << "Could not load the texture\n";
		}
		else
		{
			// Load the texture to the GPU and clear the colour buffer, also signal the UI
			m_renderer->loadTexture(texture, reader.gamma(), _type);
			m_renderer->clearBuffer();
			emit textureLoaded(location, _type);
		}
	}
}

void NGLScene::loadBRDF()
{
	// Open a file prompt and try to load MERL BRDF data
	QString location = QFileDialog::getOpenFileName(this, tr("Load BRDF Binary"), NULL, tr("Binary-files (*.binary)"));
	if(!location.isEmpty())
	{
		// Send the data to the renderer and clear the colour buffer if it was successful, also signal the UI
		if(m_renderer->loadBRDF(vBRDFLoader::loadBinary(location.toStdString())))
		{
			m_renderer->clearBuffer();
			emit brdfLoaded(location);
		}
	}
}

void NGLScene::useExampleSphere(const bool &_val)
{
	m_renderer->useExampleSphere(_val);
	m_renderer->clearBuffer();
}

void NGLScene::useBRDF(const bool &_val)
{
	m_renderer->useBRDF(_val);
	m_renderer->clearBuffer();
}

void NGLScene::useCornellEnv(const bool &_val)
{
	m_renderer->useCornellBox(_val);
	m_renderer->clearBuffer();
}

void NGLScene::changeFov(const int &_newFov)
{
	m_virtualCamera->changeFov(static_cast<float>(_newFov));
}

void NGLScene::changeFresnelCoef(const int &_newVal)
{
	m_renderer->setFresnelCoef(_newVal/100.f);
}

void NGLScene::changeFresnelPower(const int &_newVal)
{
	m_renderer->setFresnelPower(_newVal/10.f);
}
