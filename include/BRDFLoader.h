#pragma once

#include <string>
#include <memory>

class vBRDFLoader
{
public:
	static float *loadBinary(const std::string &_brdfFile);
};
