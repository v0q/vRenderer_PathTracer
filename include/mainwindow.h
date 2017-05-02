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
	void updateUITexture(const QString &_texture, const unsigned int &_type);
	void updateUIBRDF(const QString &_brdf);
  void updateSceneTree(const QString &);
	void updateUIFOV(const int &_newFov);
};
