#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>
#include "CalibrationStage.h"
#include "CalibrationTypes.h"
//********
// These are shared types (shared between app.cpp and render.cpp)
//********

// The app uses these symbolic names to tell the renderer which image to upload and draw.
// The renderer owns the actual VkImage/VkImageView/VkSampler behind each slot
// Textures are all mono for the calibration.
enum TextureSlot : int {
    TEX_START = 0,
    TEX_CALCULATING,
    TEX_SAVING,
    TEX_FAIL,
    TEX_SUCCESS,
    MAX_TEXTURES
};

//after opencv loads image (cpu intensive task, to be done on another thread)
struct DecodedImage {
    std::vector<uint8_t> pixels;
    VkFormat format = VK_FORMAT_UNDEFINED;
    int width = 0;
    int height = 0;
};

// One gaze sample to plot on the calibration result screen.
// x / y are normalized display-area coordinates (same space as Tobii reports them).
struct GazeDot {
    float x = 0.0f;
    float y = 0.0f;
    float r = 1.0f, g = 1.0f, b = 1.0f, a = 1.0f;
};

// ---------------------------------------------------------------------------
// Head positioning guide
// ---------------------------------------------------------------------------

// Half the distance between the two eyes, in track box units (~6.3 cm IPD on a ~50 cm wide track box).
// The renderer draws the target eye circles this far either side of the centre of the oval;
// the app's debug-mode simulated head uses it so the fake eyes sit exactly on those circles.
inline constexpr float HEAD_GUIDE_IDEAL_EYE_HALF_SEPARATION = 0.0625f;

// One eye, already converted to the participant's point of view (NOT raw Tobii track box coordinates).
struct HeadGuideEye {
    bool  valid = false;
    float x = 0.5f; // 0 = left edge of the track box as the participant sees it, 1 = right edge
    float y = 0.5f; // 0 = top, 1 = bottom
    float z = 0.5f; // depth: 0 = closest to the tracker / screen, 1 = furthest away. 0.5 = ideal
};

struct HeadGuide {
    HeadGuideEye left;                 // the participant's left eye
    HeadGuideEye right;                // the participant's right eye
    bool  inPosition = false;          // true -> guide is drawn green, false -> red
    float depthTolerance = 0.2f;       // +/- around z = 0.5 that counts as good (drawn as the green zone of the depth gauge)
};

// What App asks the renderer to put on screen this frame.
// The renderer translates this into Vulkan draw calls.
struct FrameScene {
    CalibrationStage mode = CalibrationStage::StartInstructions;

    // Fixation crosses to draw, in normalized display-area coordinates.
    // Calibration: just the current target. Complete: every target.
    std::vector<NormalizedPoint2D> crosses;

    // White calibration target dots, in normalized display-area coordinates.
    // Calibration: just the current target (the Complete screen uses crosses instead).
    std::vector<NormalizedPoint2D> dots;

    // Head positioning guide (HeadPositioning stage only).
    HeadGuide head;

    // Draw the dots uploaded with Renderer::setGazeDots() (Complete screen only).
    bool drawGazeDots = false;
};