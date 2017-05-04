/// \brief Main window encapsulating the UI and output window
/// \author Teemu Lindborg
/// \version 1.0
/// \date 04/05/17 Updated to NCCA Coding standard
/// Revision History :
/// Initial Version 08/12/16
/// \todo Fix scene tree, UI styling

#pragma once

#include <QMainWindow>
#include <QKeyEvent>
#include <QStandardItemModel>
#include "NGLScene.h"

namespace Ui {
  class MainWindow;
}

///
/// \brief The MainWindow class
///
class MainWindow : public QMainWindow
{
  Q_OBJECT

public:
	///
	/// \brief MainWindow Default ctor, allocates the scene and connects used signals and slots
	/// \param _parent Parent widget, passed on to the QMainWindow
	///
	explicit MainWindow(QWidget *_parent = 0);

	///
	/// \brief Default dtor, handles basic clean up
	///
  ~MainWindow();

	///
	/// \brief keyPressEvent Handles simple keypress events,
	/// \param _event Event containing the keypress data
	///
	void keyPressEvent(QKeyEvent *_event) override;

private:
	///
	/// \brief m_ui UI elements
	///
  Ui::MainWindow *m_ui;

	///
	/// \brief m_scene Main scene used to render the output to
	///
	NGLScene *m_scene;

	///
	/// \brief m_model Item model used in the scene tree to view currently used models, not in use at the moment
	///
	QStandardItemModel m_model;

public slots:
	///
	/// \brief updateUISceneTree Adds new meshes to the scene tree in the UI
	/// \param _mesh Name of the mesh
	///
	void updateUISceneTree(const QString &_mesh);

	///
	/// \brief updateUITexture Updates the path of a loaded texture to the UI
	/// \param _texture Path to the loaded texture
	/// \param _type Texture type, 0 = Diffuse, 1 = Normal, 2 = Specular
	///
	void updateUITexture(const QString &_texture, const unsigned int &_type);

	///
	/// \brief updateUIBRDF Updates the path of a loaded BRDF to the UI
	/// \param _brdf Path to the loaded BRDF binary
	///
	void updateUIBRDF(const QString &_brdf);

	///
	/// \brief updateUIHDRI Updates the path of a loaded HDRI to the UI
	/// \param _hdri Path to the loaded HDRI map
	///
	void updateUIHDRI(const QString &_hdri);

	///
	/// \brief updateUIFOV Updates the Field of View display in the UI
	/// \param _newFov Updated FOV value
	///
	void updateUIFOV(const int &_newFov);

	///
	/// \brief updateUIFresnelCoef Updates the fresnel coefficient display in the UI
	/// \param _newVal Updated fresnel coefficient
	///
	void updateUIFresnelCoef(const int &_newVal);

	///
	/// \brief updateUIFresnelPow Updates the fresnel power display in the UI
	/// \param _newVal Updated fresnel power
	///
	void updateUIFresnelPow(const int &_newVal);

	///
	/// \brief updateUIFXAASoftness Updates the FXAA softness/sharpness display in the UI
	/// \param _newVal Updated FXAA sharpness value
	///
	void updateUIFXAASoftness(const int &_newVal);

	///
	/// \brief updateUIFXAAEdgeThreshold Updates the FXAA edge threshold display in the UI
	/// \param _newVal New edge threshold
	///
	void updateUIFXAAEdgeThreshold(const int &_newVal);

	///
	/// \brief updateUIFXAASubpixQuality Updates the FXAA subpixel quality display in the UI
	/// \param _newVal Updated subpixel quality value
	///
	void updateUIFXAASubpixQuality(const int &_newVal);
};
