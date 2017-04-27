#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow() :
  m_ui(new Ui::MainWindow)
{
  QSurfaceFormat format;
  format.setSamples(4);
  format.setVersion(4, 1);
  format.setProfile(QSurfaceFormat::CoreProfile);
  format.setSwapInterval(0);
  QSurfaceFormat::setDefaultFormat(format);

  m_ui->setupUi(this);
	m_scene = new NGLScene(this);

	m_ui->m_renderWidgetLayout->addWidget(m_scene, 0, 0, 1, 1);
	connect(m_ui->m_loadMeshBtn, SIGNAL(released()), m_scene, SLOT(loadMesh()));
	connect(m_ui->m_loadDiffuseTextureBtn, SIGNAL(released()), m_scene, SLOT(loadDiffuse()));
	connect(m_ui->m_loadNormalTextureBtn, SIGNAL(released()), m_scene, SLOT(loadNormal()));
	connect(m_ui->m_loadSpecularTextureBtn, SIGNAL(released()), m_scene, SLOT(loadSpecular()));
	connect(m_ui->m_fovSlider, SIGNAL(valueChanged(int)), m_scene, SLOT(changeFov(int)));
	connect(m_ui->m_fovSlider, SIGNAL(valueChanged(int)), this, SLOT(updateUIFOV(int)));
	connect(m_scene, SIGNAL(meshLoaded(const QString &)), this, SLOT(updateSceneTree(const QString &)));

	m_model.setHorizontalHeaderItem(0, new QStandardItem("Root"));
	m_ui->m_sceneTreeView->setModel(&m_model);

//	connect(m_ui->m_hdrBtn, SIGNAL(pressed()), this, SLOT(showHideHDRMenu()));
}

MainWindow::~MainWindow()
{
	delete m_scene;
  delete m_ui;
}

void MainWindow::keyPressEvent(QKeyEvent *_event)
{
	switch (_event->key())
	{
		case Qt::Key_Escape : QApplication::exit(EXIT_SUCCESS); break;
		case Qt::Key_Return : m_scene->changeRenderChannel(); std::cout << "Triggered\n"; break;

		default: break;
	}
}

void MainWindow::updateSceneTree(const QString &_mesh)
{
  QStandardItem *mesh = new QStandardItem(_mesh);
  m_model.setItem(0, 0, mesh);
}

void MainWindow::updateUIFOV(const int &_newFov)
{
	m_ui->m_fovNumber->display(_newFov);
}

void MainWindow::showHideHDRMenu()
{
//  QSize geom = m_ui->m_hdrLayout->sizeHint();
//	m_ui->m_hdrLayout->sizeHint().setHeight(0);
	std::cout << "Triggered\n";
}
