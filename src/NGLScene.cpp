#include <QMouseEvent>
#include <QGuiApplication>
#include <ngl/ShaderLib.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "NGLScene.h"
#include <ngl/NGLInit.h>

#include "PathTracer.cuh"

NGLScene::NGLScene() : m_frame(1), m_modelPos(ngl::Vec3(0.0f, 0.0f, 0.0f))
{
  // re-size the widget to that of the parent (in this case the GLFrame passed in on construction)
  setTitle("Blank NGL");
}

NGLScene::~NGLScene()
{
	cudaFree(m_colorArray);
	cudaGraphicsUnregisterResource(m_cudaGLTextureBuffer);
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
  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);			   // Grey Background
  // enable depth testing for drawing
  glEnable(GL_DEPTH_TEST);
  // enable multisampling for smoother drawing
  glEnable(GL_MULTISAMPLE);

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

	// Unbind the texture

	validateCuda(cudaGraphicsGLRegisterImage(&m_cudaGLTextureBuffer, m_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	unsigned int sz = width()*height();
	validateCuda(cudaMalloc(&m_colorArray, sizeof(float3)*sz));
	cu_fillFloat3(m_colorArray, make_float3(0.0f, 0.0f, 0.0f), sz);
//	validateCuda(cudaMemcpy(m_colorArray, &zeros, width()*height(), cudaMemcpyHostToDevice));

	glBindTexture(GL_TEXTURE_2D, 0);

	startTimer(10);
}

void NGLScene::timerEvent(QTimerEvent *_event)
{
	update();
}

void NGLScene::paintGL()
{
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	static float t = 0;
	t += 0.1f;
  // clear the screen and depth buffer
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0,0,m_win.width,m_win.height);

	ngl::Mat4 rot;
	rot.rotateX(m_win.spinXFace/25.f);
	rot.rotateY(m_win.spinYFace/25.f);

	ngl::Vec4 cam = rot*ngl::Vec4(50, 52, 295.6);
	ngl::Vec4 dir = ngl::Vec4(m_modelPos.m_x/5., m_modelPos.m_y/5., 0.0f) + ngl::Vec4(0, -0.042612, -1);
	dir = dir.normalize();

	validateCuda(cudaGraphicsMapResources(1, &m_cudaGLTextureBuffer));
	validateCuda(cudaGraphicsSubResourceGetMappedArray(&m_cudaImgArray, m_cudaGLTextureBuffer, 0, 0));

	cudaResourceDesc wdsc;
	wdsc.resType = cudaResourceTypeArray;
	wdsc.res.array.array = m_cudaImgArray;
	cudaSurfaceObject_t writeSurface;
	validateCuda(cudaCreateSurfaceObject(&writeSurface, &wdsc));
	cu_ModifyTexture(writeSurface,
									 m_colorArray,
									 make_float3(cam.m_x, cam.m_y, cam.m_z),
									 make_float3(dir.m_x, dir.m_y, dir.m_z),
									 width(), height(), m_frame++, std::chrono::duration_cast<std::chrono::milliseconds>(t1.time_since_epoch()).count());
	validateCuda(cudaDestroySurfaceObject(writeSurface));
	validateCuda(cudaGraphicsUnmapResources(1, &m_cudaGLTextureBuffer));
	validateCuda(cudaStreamSynchronize(0));

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
//	glDisableVertexAttribArray(uvLocation);

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

	std::cout << "Took " << duration << "ms to path trace the scene with 2048 SPP\n";
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
  default : break;
  }
  // finally update the GLWindow and re-draw

    update();
}
