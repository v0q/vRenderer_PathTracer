#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *_parent) :
	QMainWindow(_parent),
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

	// Scene/mesh loading
	connect(m_ui->m_loadMeshBtn, SIGNAL(released()), m_scene, SLOT(loadMesh()));

	// Environment
	connect(m_ui->m_loadHDRBtn, SIGNAL(released()), m_scene, SLOT(loadHDR()));
	connect(m_ui->m_envCornell, SIGNAL(toggled(bool)), m_scene, SLOT(useCornellEnv(const bool &)));

	// Textures
	connect(m_ui->m_loadDiffuseTextureBtn, SIGNAL(released()), m_scene, SLOT(loadDiffuse()));
	connect(m_ui->m_loadNormalTextureBtn, SIGNAL(released()), m_scene, SLOT(loadNormal()));
	connect(m_ui->m_loadSpecularTextureBtn, SIGNAL(released()), m_scene, SLOT(loadSpecular()));

	// Example sphere
	connect(m_ui->m_useSphere, SIGNAL(toggled(bool)), m_scene, SLOT(useExampleSphere(bool)));

	// BRDF
	connect(m_ui->m_useBRDF, SIGNAL(toggled(bool)), m_scene, SLOT(useBRDF(bool)));
	connect(m_ui->m_loadBRDFBtn, SIGNAL(released()), m_scene, SLOT(loadBRDF()));

	// Camera
	connect(m_ui->m_fovSlider, SIGNAL(valueChanged(int)), m_scene, SLOT(changeFov(int)));
	connect(m_ui->m_fovSlider, SIGNAL(valueChanged(int)), this, SLOT(updateUIFOV(int)));

	// Fresnel
	connect(m_ui->m_fresnelSlider, SIGNAL(valueChanged(int)), m_scene, SLOT(changeFresnelCoef(const int &)));
	connect(m_ui->m_fresnelSlider, SIGNAL(valueChanged(int)), this, SLOT(updateUIFresnelCoef(int)));

	connect(m_ui->m_fresnelPowSlider, SIGNAL(valueChanged(int)), m_scene, SLOT(changeFresnelPower(const int &)));
	connect(m_ui->m_fresnelPowSlider, SIGNAL(valueChanged(int)), this, SLOT(updateUIFresnelPow(int)));

	// FXAA
	connect(m_ui->m_useFXAA, SIGNAL(toggled(bool)), m_scene, SLOT(toggleFXAA(const bool &)));

	connect(m_ui->m_fxaaSoftnessSlider, SIGNAL(valueChanged(int)), m_scene, SLOT(fxaaSharpness(const int &)));
	connect(m_ui->m_fxaaSoftnessSlider, SIGNAL(valueChanged(int)), this, SLOT(updateUIFXAASoftness(const int &)));

	connect(m_ui->m_fxaaSubpixEdgeThresholdSlider, SIGNAL(valueChanged(int)), m_scene, SLOT(fxaaEdgeThreshold(const int &)));
	connect(m_ui->m_fxaaSubpixEdgeThresholdSlider, SIGNAL(valueChanged(int)), this, SLOT(updateUIFXAAEdgeThreshold(const int &)));

	connect(m_ui->m_fxaaSubpixQualitySlider, SIGNAL(valueChanged(int)), m_scene, SLOT(fxaaSubpixQuality(const int &)));
	connect(m_ui->m_fxaaSubpixQualitySlider, SIGNAL(valueChanged(int)), this, SLOT(updateUIFXAASubpixQuality(const int &)));

	connect(m_ui->m_fxaaSoftnessSlider, SIGNAL(valueChanged(int)), m_scene, SLOT(fxaaSharpness(const int &)));
	connect(m_ui->m_fxaaSoftnessSlider, SIGNAL(valueChanged(int)), this, SLOT(updateUIFXAASoftness(const int &)));

	// Signals
	connect(m_scene, SIGNAL(textureLoaded(const QString &, const unsigned int &)), this, SLOT(updateUITexture(const QString &, const unsigned int &)));
	connect(m_scene, SIGNAL(meshLoaded(const QString &)), this, SLOT(updateUISceneTree(const QString &)));
	connect(m_scene, SIGNAL(brdfLoaded(const QString &)), this, SLOT(updateUIBRDF(const QString &)));
	connect(m_scene, SIGNAL(HDRILoaded(const QString &)), this, SLOT(updateUIHDRI(const QString &)));

	m_model.setHorizontalHeaderItem(0, new QStandardItem("Root"));
	m_ui->m_sceneTreeView->setModel(&m_model);

	// For whatever reason not every keypress triggers without this
	this->setFocusPolicy(Qt::StrongFocus);
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
		case Qt::Key_Escape : QGuiApplication::exit(EXIT_SUCCESS); break;
		case Qt::Key_Return : m_scene->changeRenderChannel(); break;

		default: break;
	}
}

void MainWindow::updateUITexture(const QString &_texture, const unsigned int &_type)
{
	switch(_type)
	{
		case 0:
			m_ui->m_diffuseLabel->setText(_texture);
		break;
		case 1:
			m_ui->m_normalLabel->setText(_texture);
		break;
		case 2:
			m_ui->m_specularLabel->setText(_texture);
		break;

		default: break;
	}
}

void MainWindow::updateUISceneTree(const QString &_mesh)
{
	QStandardItem *mesh = new QStandardItem(_mesh);
	m_model.setItem(0, 0, mesh);
}

void MainWindow::updateUIBRDF(const QString &_brdf)
{
	m_ui->m_brdfLocation->setText(_brdf);
}

void MainWindow::updateUIHDRI(const QString &_hdri)
{
	m_ui->m_hdrLabel->setText(_hdri);
}

void MainWindow::updateUIFOV(const int &_newFov)
{
	m_ui->m_fovNumber->display(_newFov);
}

void MainWindow::updateUIFresnelCoef(const int &_newVal)
{
	m_ui->m_fresnelNumber->display(_newVal/100.f);
}

void MainWindow::updateUIFresnelPow(const int &_newVal)
{
	m_ui->m_fresnelPowNumber->display(_newVal/10.f);
}

void MainWindow::updateUIFXAASoftness(const int &_newVal)
{
	m_ui->m_fxaaSoftnessNum->display(_newVal/100.f);
}

void MainWindow::updateUIFXAAEdgeThreshold(const int &_newVal)
{
	m_ui->m_fxaaSubpixEdgeThresholdNum->display(_newVal/1000.f);
}

void MainWindow::updateUIFXAASubpixQuality(const int &_newVal)
{
	m_ui->m_fxaaSubpixQualityNum->display(_newVal/100.f);
}

void MainWindow::showHideHDRMenu()
{
//  QSize geom = m_ui->m_hdrLayout->sizeHint();
//	m_ui->m_hdrLayout->sizeHint().setHeight(0);
	std::cout << "Triggered\n";
}
