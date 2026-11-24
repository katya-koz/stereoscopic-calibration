#pragma once
#include <Windows.h>
#include <string>
#include <algorithm>
#include <random>
#include "app.h"
#include <filesystem>
#include "tobii_research_streams.h"
#include <cmath>
#include <vector>
#include <math.h>
#include <numbers>
#include <limits>  
#include <sstream>
#include <iostream>
#include <cctype>
namespace Utils
{

	static void FatalError(const std::string& message)
	{
		const auto result = MessageBoxA(
			nullptr,
			message.c_str(),
			"Fatal Error",
			MB_OK | MB_ICONERROR | MB_TOPMOST
		);

		if (result == IDOK)
		{
			exit(1);
		}
	}
	



	static std::filesystem::path getExecutableDirectory()
	{
		char buffer[MAX_PATH];

		DWORD length = GetModuleFileNameA(
			nullptr,
			buffer,
			MAX_PATH
		);

		if (length == 0) {
			throw std::runtime_error("Failed to get executable path.");
		}

		return std::filesystem::path(buffer).parent_path();
	}


}