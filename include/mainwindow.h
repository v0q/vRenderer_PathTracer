#ifndef MAINWINDOW_H
#define MAINWINDOW_H

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
  explicit MainWindow();
  ~MainWindow();

private:
  void keyPressEvent(QKeyEvent *_event) override;

  Ui::MainWindow *m_ui;
	NGLScene *m_scene;

  QStandardItemModel m_model;

public slots:
  void showHideHDRMenu();
  void updateSceneTree(const QString &);
	void updateUIFOV(const int &_newFov);
};

#endif // MAINWINDOW_H
