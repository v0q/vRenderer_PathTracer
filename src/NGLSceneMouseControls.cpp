#include "NGLScene.h"
#include <ngl/Mat4.h>
#include <ngl/Vec4.h>
#include <QMouseEvent>

#ifdef __VRENDERER_CUDA__
  #include <cuda_runtime.h>
  #include "PathTracer.cuh"
#elif __VRENDERER_OPENCL__

#endif

//----------------------------------------------------------------------------------------------------------------------
void NGLScene::mouseMoveEvent( QMouseEvent* _event )
{
  // note the method buttons() is the button state when event was called
  // that is different from button() which is used to check which button was
  // pressed when the mousePress/Release event is generated
  if ( m_win.rotate && _event->buttons() == Qt::LeftButton )
  {
    int diffx = _event->x() - m_win.origX;
    int diffy = _event->y() - m_win.origY;
    m_win.spinXFace += static_cast<int>( 0.5f * diffy );
    m_win.spinYFace += static_cast<int>( 0.5f * diffx );
    m_win.origX = _event->x();
    m_win.origY = _event->y();
		m_frame = 1;

		ngl::Mat4 rot;
		rot.rotateX(m_win.spinXFace/25.f);
		rot.rotateY(m_win.spinYFace/25.f);

		ngl::Vec4 cam = rot*ngl::Vec4(50, 52, 295.6);

#ifdef __VRENDERER_CUDA__
		validateCuda(cudaMemcpy(m_camera, &cam.m_openGL[0], sizeof(float3), cudaMemcpyHostToDevice));
		cu_fillFloat3(m_colorArray, make_float3(0.0f, 0.0f, 0.0f), width()*height());
		cudaDeviceSynchronize();
#elif __VRENDERER_OPENCL__

#endif

    update();
  }
  // right mouse translate code
  else if ( m_win.translate && _event->buttons() == Qt::RightButton )
  {
		int diffX      = static_cast<int>( _event->x() - m_win.origXPos );
		int diffY      = static_cast<int>( _event->y() - m_win.origYPos );
    m_win.origXPos = _event->x();
    m_win.origYPos = _event->y();
		m_modelPos.m_x += INCREMENT * diffX;
		m_modelPos.m_y -= INCREMENT * diffY;
		m_frame = 1;

		ngl::Vec4 dir = ngl::Vec4(m_modelPos.m_x/5., m_modelPos.m_y/5., 0.0f) + ngl::Vec4(0, -0.042612, -1);
		dir = dir.normalize();

#ifdef __VRENDERER_CUDA__
		validateCuda(cudaMemcpy(m_camdir, &dir.m_openGL[0], sizeof(float3), cudaMemcpyHostToDevice));
		cu_fillFloat3(m_colorArray, make_float3(0.0f, 0.0f, 0.0f), width()*height());
		cudaDeviceSynchronize();
#elif __VRENDERER_OPENCL__

#endif

    update();
  }
}


//----------------------------------------------------------------------------------------------------------------------
void NGLScene::mousePressEvent( QMouseEvent* _event )
{
  // that method is called when the mouse button is pressed in this case we
  // store the value where the maouse was clicked (x,y) and set the Rotate flag to true
  if ( _event->button() == Qt::LeftButton )
  {
    m_win.origX  = _event->x();
    m_win.origY  = _event->y();
    m_win.rotate = true;
  }
  // right mouse translate mode
  else if ( _event->button() == Qt::RightButton )
  {
    m_win.origXPos  = _event->x();
    m_win.origYPos  = _event->y();
    m_win.translate = true;
  }
}

//----------------------------------------------------------------------------------------------------------------------
void NGLScene::mouseReleaseEvent( QMouseEvent* _event )
{
  // that event is called when the mouse button is released
  // we then set Rotate to false
  if ( _event->button() == Qt::LeftButton )
  {
    m_win.rotate = false;
  }
  // right mouse translate mode
  if ( _event->button() == Qt::RightButton )
  {
    m_win.translate = false;
  }
}

//----------------------------------------------------------------------------------------------------------------------
void NGLScene::wheelEvent( QWheelEvent* _event )
{

  // check the diff of the wheel position (0 means no change)
  if ( _event->delta() > 0 )
  {
    m_modelPos.m_z += ZOOM;
  }
  else if ( _event->delta() < 0 )
  {
    m_modelPos.m_z -= ZOOM;
  }
  update();
}
