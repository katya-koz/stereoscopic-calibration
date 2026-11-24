#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <string>

namespace fs = std::filesystem;

/// <summary>
/// The screen's position as the eye tracker sees it: three corners of the active display area, in millimetres,
/// in the tracker's own coordinate system (origin at the centre of the tracker's front surface, +x towards the
/// user's right, +y up, +z towards the user) 
/// </summary>
struct DisplayAreaConfig {
    std::array<float, 3> topLeft{};
    std::array<float, 3> topRight{};
    std::array<float, 3> bottomLeft{};
};

/// <summary>
/// App settings, loaded from config.json.
/// </summary>
struct Config {
    // ---- display area -------------------------------------------------------------------------
    // if present, the app writes this display area to the tracker at every startup 
    // if absent, the app leaves the tracker's stored display area alone and just prints it 
    std::optional<DisplayAreaConfig> displayArea;

    // ---- calibration region -------------------------------------------------------------------
    // Size (mm) of the rectangle, centred on the monitor, that the calibration targets are spread over.
    // 0 (or missing) = use the whole screen on that axis.
    double calibrationWidthMm = 0.0;
    double calibrationHeightMm = 0.0;

    // Where the calibration csv is written. Defaults to <exe dir>/calibration.
    fs::path outputDirectory;

    // Set at runtime by the popup shown at startup (already sanitized), not read from the config file.
    std::string participantId;

    /// <summary> Loads and validates config.json. Returns false (after reporting why) if it can't be used. </summary>
    bool loadConfig(const std::string& configPath);

    std::string toString() const;
};