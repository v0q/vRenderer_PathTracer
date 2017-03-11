#include <QMouseEvent>
#include <QGuiApplication>
#include <ngl/ShaderLib.h>
#include <iostream>
#include <vector>
#include <assert.h>
#include <chrono>
#include <OpenEXR/ImfRgba.h>
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImathBox.h>
#include <OpenColorIO/OpenColorIO.h>

#include "NGLScene.h"
#include <ngl/NGLInit.h>

#include "MeshLoader.h"
#ifdef __VRENDERER_CUDA__
	#include "vRendererCuda.h"
#elif __VRENDERER_OPENCL__
	#include "vRendererCL.h"
#endif

namespace OCIO = OCIO_NAMESPACE;

NGLScene::NGLScene() :
  m_modelPos(ngl::Vec3(0.0f, 0.0f, 0.0f))
{
  // re-size the widget to that of the parent (in this case the GLFrame passed in on construction)
	setTitle("vRenderer");
	m_renderTexture = false;
	m_fpsTimer = startTimer(0);
	m_fps = 0;
	m_frames = 0;
	m_timer.start();
}

NGLScene::~NGLScene()
{
  glDeleteBuffers(1, &m_vbo);
  std::cout<<"Shutting down NGL, removing VAO's and Shaders\n";
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

#ifdef __VRENDERER_CUDA__
  m_renderer.reset(new vRendererCuda);
#elif __VRENDERER_OPENCL__
  m_renderer.reset(new vRendererCL);
#endif

  m_renderer->init((unsigned int)width(), (unsigned int)height());

	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_MULTISAMPLE);

	ngl::ShaderLib *shader = ngl::ShaderLib::instance();

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

	shader->use("Screen Quad");

	glGenVertexArrays(1, &m_vao);
	glBindVertexArray(m_vao);

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

//	OCIO::ConstConfigRcPtr config = OCIO::GetCurrentConfig();

//	const char *display = config->getDefaultDisplay();
//	const char *view = config->getDefaultView(display);
//	const char *transform = config->getDisplayColorSpaceName(display, view);

//	OCIO::ConstProcessorRcPtr processor = config->getProcessor(OCIO::ROLE_SCENE_LINEAR, transform);

//	std::cout << display << " " << view << " " << transform << "\n";

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
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Create texture data (4-component unsigned byte)
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width(), height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

  m_renderer->registerTextureBuffer(m_texture);

  // Unbind the texture
	glBindTexture(GL_TEXTURE_2D, 0);

	m_text.reset(new ngl::Text(QFont("Arial", 12)));
	m_text->setScreenSize(width(), height());

//	m_renderer->initMesh(vMeshLoader::loadMesh("models/cube.obj"));
//  m_renderer->initMesh(vMeshLoader::loadMesh("models/icosahedron.obj"));
//  m_renderer->initMesh(vMeshLoader::loadMesh("models/lowpolytree.obj"));
//  m_renderer->initMesh(vMeshLoader::loadMesh("models/bunny.obj"));
  m_renderer->initMesh(vMeshLoader::loadMesh("models/dragon_vrip_res2.ply"));

  Imf::Rgba *pixelBuffer;
  try
  {
    Imf::RgbaInputFile in("hdr/test.exr");
    Imath::Box2i win = in.dataWindow();

    Imath::V2i dim(win.max.x - win.min.x + 1,
                   win.max.y - win.min.y + 1);

    pixelBuffer = new Imf::Rgba[dim.x *dim.y];

    int dx = win.min.x;
    int dy = win.min.y;

    in.setFrameBuffer(pixelBuffer - dx - dy * dim.x, 1, dim.x);
    in.readPixels(win.min.y, win.max.y);

    m_renderer->initHDR(pixelBuffer, dim.x, dim.y);
  }
  catch (Iex::BaseExc &e)
  {
    std::cerr << e.what() << "\n";
    exit(0);
  }
}

void NGLScene::timerEvent(QTimerEvent *_event)
{
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

	static float t = 0;
	t += 0.1f;
  // clear the screen and depth buffer
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0,0,m_win.width,m_win.height);

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	m_renderer->render();

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	ngl::ShaderLib *shader = ngl::ShaderLib::instance();
	shader->use("Screen Quad");

	glBindVertexArray(m_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

	GLuint posLocation = shader->getAttribLocation("Screen Quad", "a_Position");
	GLuint uvLocation = shader->getAttribLocation("Screen Quad", "a_FragCoord");

	glEnableVertexAttribArray(posLocation);
	glEnableVertexAttribArray(uvLocation);

	glVertexAttribPointer(posLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribPointer(uvLocation, 2, GL_FLOAT, GL_FALSE, 0, (GLvoid *)(12*sizeof(float)));

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_texture);

	shader->setRegisteredUniform1i("u_ptResult", 0);

	glDrawArrays(GL_TRIANGLES, 0, 6);

	// Clean up
	shader->useNullProgram();
	glDisableVertexAttribArray(posLocation);
	glDisableVertexAttribArray(uvLocation);

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	if(m_renderTexture) {
		// TODO
		m_renderTexture = false;
	}

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
	m_text->setColour(1, 1, 1);
	QString text = QString("%1 fps").arg(m_fps);
	m_text->renderText(10, 20, text);
	text = QString("Render time/frame: %1ms").arg(duration);
	m_text->renderText(10, 40, text);
	text = QString("%1 samples per pixel").arg(m_renderer->getFrameCount() * 2);
	m_text->renderText(10, 60, text);
	++m_frames;
}

//----------------------------------------------------------------------------------------------------------------------

void NGLScene::keyPressEvent(QKeyEvent *_event)
{
  // this method is called every time the main window recives a key event.
  // we then switch on the key value and set the camera in the GLWindow
  switch (_event->key())
  {
  // escape key to quite
  case Qt::Key_Escape : QGuiApplication::exit(EXIT_SUCCESS); break;
  case Qt::Key_Space :
      m_win.spinXFace=0;
      m_win.spinYFace=0;
      m_modelPos.set(ngl::Vec3::zero());

  break;
	case Qt::Key_R: m_renderTexture = true; break;
  default : break;
  }
  // finally update the GLWindow and re-draw

		update();
}
