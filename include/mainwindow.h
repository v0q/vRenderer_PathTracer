#pragma once

#include <QMainWindow>
#include <QKeyEvent>
#include <QStandardItemModel>
#include "NGLScene.h"

namespace Ui {
  class MainWindow;
}

class MainWindow : public QMainWindow
{
  Q_OBJECT

public:
	explicit MainWindow(QWidget *_parent = 0);
  ~MainWindow();
	void keyPressEvent(QKeyEvent *_event) override;

private:
  Ui::MainWindow *m_ui;
	NGLScene *m_scene;

	QStandardItemModel m_model;

public slots:
  void showHideHDRMenu();
	void updateUISceneTree(const QString &);
	void updateUITexture(const QString &_texture, const unsigned int &_type);
	void updateUIBRDF(const QString &_brdf);
	void updateUIHDRI(const QString &_hdri);
	void updateUIFOV(const int &_newFov);
	void updateUIFresnelCoef(const int &_newVal);
	void updateUIFresnelPow(const int &_newVal);
	void updateUIFXAASoftness(const int &_newVal);
	void updateUIFXAAEdgeThreshold(const int &_newVal);
	void updateUIFXAASubpixQuality(const int &_newVal);
};
