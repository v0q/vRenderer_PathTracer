///
/// \file BRDFLoader.h
/// \brief Loads MERL BRDF binary data to RBG format. Data loading from the binary is based on
///				 the MERL 100 code available at http://people.csail.mit.edu/wojciech/BRDFDatabase/code/BRDFRead.cpp
/// \authors Teemu Lindborg
/// \version 1.0
/// \date 04/05/17 Updated to NCCA Coding standard
/// Revision History :
/// Initial Version 03/05/17
/// \todo Use the double data instead
///

#pragma once

#include <string>
#include <memory>

///
/// \brief The vBRDFLoader class Currently only handles simple loading of the brdf binary data
///
class vBRDFLoader
{
public:
	///
	/// \brief loadBinary Reads in the BRDF binary file and returns a float pointer to the RGB data
	/// \param _brdfFile File to read in
	/// \return Floating point pointer to RBG data of the measured BRDF
	///
	static float *loadBinary(const std::string &_brdfFile);
};
