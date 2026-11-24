#include "config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "Utils.h"

using json = nlohmann::json;

namespace {

    // Reads j[key] into `out` if the key exists. Returns false only if it exists but is not a number.
    bool readOptionalNumber(const json& j, const char* key, double& out) {
        const auto it = j.find(key);
        if (it == j.end()) return true;

        if (!it->is_number()) {
            std::cerr << "[Config] \"" << key << "\" must be a number\n";
            return false;
        }
        out = it->get<double>();
        return true;
    }

    // Same, for sizes that must be strictly positive when they are given.
    bool readOptionalPositiveNumber(const json& j, const char* key, double& out) {
        if (!readOptionalNumber(j, key, out)) return false;

        if (j.contains(key) && out <= 0.0) {
            std::cerr << "[Config] \"" << key << "\" must be greater than 0 (or left out to use the full screen)\n";
            return false;
        }
        return true;
    }

    // Reads obj[key] as [x, y, z] (millimetres).
    bool readPoint3(const json& obj, const char* key, std::array<float, 3>& out) {
        const auto it = obj.find(key);
        if (it == obj.end()) {
            std::cerr << "[Config] \"Display Area (mm)\" is missing \"" << key << "\"\n";
            return false;
        }
        if (!it->is_array() || it->size() != 3) {
            std::cerr << "[Config] \"Display Area (mm)\" -> \"" << key << "\" must be three numbers: [x, y, z]\n";
            return false;
        }
        for (size_t i = 0; i < 3; ++i) {
            const json& v = (*it)[i];
            if (!v.is_number()) {
                std::cerr << "[Config] \"Display Area (mm)\" -> \"" << key << "\" must only contain numbers\n";
                return false;
            }
            out[i] = v.get<float>();
        }
        return true;
    }

} // namespace

bool Config::loadConfig(const std::string& configPath) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        std::cerr << "[Config] Could not open config file: " << configPath << "\n";
        return false;
    }

    json j;
    try {
        // allow // and /* */ comments, so settings can be commented out while experimenting
        j = json::parse(file, nullptr, /*allow_exceptions*/ true, /*ignore_comments*/ true);
    }
    catch (const json::parse_error& e) {
        std::cerr << "[Config] JSON parse error: " << e.what() << "\n";
        return false;
    }

    if (!j.is_object()) {
        std::cerr << "[Config] config file must contain a JSON object\n";
        return false;
    }

    // everything below is optional, with the defaults from the Config struct

    // display area (where the screen is, relative to the eye tracker)
    if (j.contains("Display Area (mm)")) {
        const json& da = j["Display Area (mm)"];
        if (!da.is_object()) {
            std::cerr << "[Config] \"Display Area (mm)\" must be an object with \"Top Left\", \"Top Right\" "
                "and \"Bottom Left\"\n";
            return false;
        }

        DisplayAreaConfig area;
        if (!readPoint3(da, "Top Left", area.topLeft)) return false;
        if (!readPoint3(da, "Top Right", area.topRight)) return false;
        if (!readPoint3(da, "Bottom Left", area.bottomLeft)) return false;

        displayArea = area;
    }

    // calibration region
    if (!readOptionalPositiveNumber(j, "Calibration Width (mm)", calibrationWidthMm)) return false;
    if (!readOptionalPositiveNumber(j, "Calibration Height (mm)", calibrationHeightMm)) return false;

    // output directory
    if (j.contains("Output Directory")) {
        if (!j["Output Directory"].is_string()) {
            std::cerr << "[Config] \"Output Directory\" must be a string\n";
            return false;
        }
        outputDirectory = j["Output Directory"].get<std::string>();
    }
    if (outputDirectory.empty()) {
        outputDirectory = Utils::getExecutableDirectory() / "calibration";
    }

    // create it now, so a bad path is reported at startup instead of after a whole calibration
    std::error_code ec;
    fs::create_directories(outputDirectory, ec);
    if (ec || !fs::is_directory(outputDirectory)) {
        Utils::FatalError("[Config] Output directory does not exist and could not be created: " +
            outputDirectory.string());
        return false;
    }

    return true;
}

std::string Config::toString() const
{
    std::ostringstream out;

    out << std::boolalpha << std::fixed << std::setprecision(1);

    out << "========== CONFIG ==========\n";
    out << "outputDirectory: " << outputDirectory.string() << '\n';

    out << "calibrationWidthMm: ";
    if (calibrationWidthMm > 0.0) out << calibrationWidthMm; else out << "full screen";
    out << '\n';

    out << "calibrationHeightMm: ";
    if (calibrationHeightMm > 0.0) out << calibrationHeightMm; else out << "full screen";
    out << '\n';

    out << "displayArea: " << (displayArea ? "from config (written to the tracker at startup)"
        : "not in config (using the one stored on the tracker)") << '\n';
    out << "============================";

    return out.str();
}