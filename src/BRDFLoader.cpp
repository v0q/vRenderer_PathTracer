///
/// \file BRDFLoader.cpp
/// \brief Loads MERL BRDF binary data
///

#include <fstream>
#include <iostream>

#include "BRDFLoader.h"

#define BRDF_SAMPLING_RES_THETA_H       90
#define BRDF_SAMPLING_RES_THETA_D       90
#define BRDF_SAMPLING_RES_PHI_D         360

float* vBRDFLoader::loadBinary(const std::string &_brdfFile)
{
	std::ifstream file(_brdfFile, std::ios::binary);

	// Read the dimensions from the binary
	int dims[3];
	file.read(reinterpret_cast<char *>(dims), 3*sizeof(int));

	// Make sure the dimensions match the expected
	int n = dims[0] * dims[1] * dims[2];
	if (n != BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D * BRDF_SAMPLING_RES_PHI_D / 2)
	{
		std::cerr << "Dimensions don't match\n";
		file.close();
		return nullptr;
	}

	// Create a buffer for the sampled values and read them in
	double *brdf = new double[3 * n];
	file.read(reinterpret_cast<char *>(brdf), 3*n*sizeof(double));

	// As we don't really need double precision, read the values to a float buffer instead
	float *vals = new float[3 * n];
	for(int i = 0; i < n; ++i)
	{
		vals[i*3] = brdf[i*3];
		vals[i*3 + 1] = brdf[i*3 + 1];
		vals[i*3 + 2] = brdf[i*3 + 2];
	}

	// Clean the double buffer and return the floating point data
	delete [] brdf;
	file.close();

	return vals;
}
