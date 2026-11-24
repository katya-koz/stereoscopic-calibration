#pragma once
#include <vector>

// A position in normalized display-area coordinates, exactly as Tobii uses them:
// (0,0) = top-left of the physical monitor, (1,1) = bottom-right.
struct NormalizedPoint2D {
    float x = 0.5f;
    float y = 0.5f;
};

// mirrors TobiiResearchCalibrationEyeValidity (keeps tobii headers out of the renderer class)
enum class EyeValidity {
    Invalid, // sample not valid, not used
    ValidNotUsed, // sample valid, but discarded by the calibration
    ValidUsed // sample valid and used by the calibration
};

struct EyeSample {
    float x = 0.0f;
    float y = 0.0f;
    EyeValidity validity = EyeValidity::Invalid;
};

struct CalibrationSample {
    EyeSample left;
    EyeSample right;
};

// everything Tobii collected for one fixation target.
struct CalibrationPointData {
    NormalizedPoint2D target;
    std::vector<CalibrationSample> samples;
};