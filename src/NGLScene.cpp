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

#include <OpenImageIO/imageio.h>
//#include <OpenColorIO/OpenColorIO.h>

#include "NGLScene.h"
#include <ngl/NGLInit.h>

#include "MeshLoader.h"
#include "BRDFLoader.h"
#include "hdrloader.h"

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
	m_yaw(0.f),
	m_pitch(0.f),
	m_fxaaSharpness(0.5f),
	m_fxaaSubpixQuality(0.75f),
	m_fxaaEdgeThreshold(0.166f),
	m_brdf(nullptr)
{
  // re-size the widget to that of the parent (in this case the GLFrame passed in on construction)
	m_renderTexture = false;
	m_fpsTimer = startTimer(0);
	m_fps = 0;
	m_frames = 0;
	m_timer.start();

	m_virtualCamera = new Camera;
}

NGLScene::~NGLScene()
{
	delete m_virtualCamera;
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
	m_renderer->setCamera(m_virtualCamera);

	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_MULTISAMPLE);

	ngl::ShaderLib *shader = ngl::ShaderLib::instance();

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

  m_renderer->registerTextureBuffer(m_texture);
	m_renderer->registerDepthBuffer(m_depthTexture);

  // Unbind the texture
	glBindTexture(GL_TEXTURE_2D, 0);

	m_text.reset(new ngl::Text(QFont("Arial", 12)));
	m_text->setScreenSize(width(), height());

//	m_renderer->initMesh(vMeshLoader::loadMesh("models/cube.obj"));
//  m_renderer->initMesh(vMeshLoader::loadMesh("models/icosahedron.obj"));
//	m_renderer->initMesh(vMeshLoader::loadMesh("models/dragon_vrip_res2.obj"));
//	m_renderer->initMesh(vMeshLoader::loadMesh("models/happy_buddha.obj"));
	m_renderer->initMesh(vMeshLoader::loadMesh("models/matt.obj"));
//	m_renderer->initMesh(vMeshLoader::loadMesh("models/adam_mask.obj"));
//	m_renderer->initMesh(vMeshLoader::loadMesh("models/adam_head.obj"));
//	m_renderer->initMesh(vMeshLoader::loadMesh("models/sebastian_head.obj"));

//	HDRLoaderResult result;
//	if(!HDRLoader::load("hdr/Arches_E_PineTree_3k.hdr", result))
//	{
//		std::cerr << "Failed to load the HDR. Exiting...\n";
//		exit(0);
//	}

//	m_renderer->initHDR(result.cols, result.width, result.height);
//	m_renderer->loadBRDF(vBRDFLoader::loadBinary("brdf/alum-bronze.binary"));
//	m_renderer->loadBRDF(vBRDFLoader::loadBinary("brdf/red-fabric.binary"));
//	m_renderer->loadBRDF(vBRDFLoader::loadBinary("brdf/red-metallic-paint.binary"));
//	m_renderer->loadBRDF(vBRDFLoader::loadBinary("brdf/cherry-235.binary"));
//	m_renderer->loadBRDF(vBRDFLoader::loadBinary("brdf/yellow-matte-plastic.binary"));

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

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, m_depthTexture);
	shader->setRegisteredUniform1i("u_ptDepth", 1);
	shader->setRegisteredUniform1i("u_channel", m_renderChannel);

	shader->setRegisteredUniform2f("u_screenDim", width(), height());
	shader->setRegisteredUniform2f("u_invScreenDim", 1.f/width(), 1.f/height());

	if(m_fxaaEnabled)
	{
		shader->setRegisteredUniform1f("u_SubPixQuality", m_fxaaSubpixQuality);
		shader->setRegisteredUniform1f("u_EdgeThreshold", m_fxaaEdgeThreshold);
		shader->setRegisteredUniform1f("u_Sharpness", m_fxaaSharpness);
	}

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
  QString location = QFileDialog::getOpenFileName(this, tr("Load mesh"), NULL, tr("3d models (*.obj *.ply)"));
	if(!location.isEmpty())
	{
		vMeshData mesh = vMeshLoader::loadMesh(location.toStdString());
		m_renderer->initMesh(mesh);
		m_renderer->clearBuffer();

		emit meshLoaded(mesh.m_name);
	}
}

void NGLScene::loadHDR()
{
	QString location = QFileDialog::getOpenFileName(this, tr("Load texture"), NULL, tr("EXR-files (*.exr)"));
	if(!location.isEmpty())
	{
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
		}
		catch (Iex::BaseExc &e)
		{
			std::cerr << e.what() << "\n";
		}
	}
}

void NGLScene::loadTexture(const unsigned int &_type)
{
	QString location = QFileDialog::getOpenFileName(this, tr("Load texture"), NULL, tr("Image files (*.jpg *.jpeg *.tif *.tiff *.png)"));
	if(!location.isEmpty())
	{
		QImageReader reader(location);
		QImage texture(location);
		if(texture.isNull())
		{
			std::cerr << "Could not load the texture\n";
		}
		else
		{
			m_renderer->loadTexture(texture, reader.gamma(), _type);
		}

		m_renderer->clearBuffer();
		emit textureLoaded(location, _type);
	}
}

void NGLScene::loadBRDF()
{
	QString location = QFileDialog::getOpenFileName(this, tr("Load BRDF Binary"), NULL, tr("Binary-files (*.binary)"));
	if(!location.isEmpty())
	{
		if(m_brdf)
		{
			delete [] m_brdf;
		}

		if((m_brdf = vBRDFLoader::loadBinary(location.toStdString())))
		{
			m_renderer->loadBRDF(m_brdf);
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
