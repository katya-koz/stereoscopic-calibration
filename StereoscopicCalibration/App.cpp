#include "app.h"
#include <cmath>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <mmsystem.h>
#include "render.h" 
#include "ParticipantPrompt.h"
#include "tobii_research.h"
#include "tobii_research_calibration.h"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <thread>
#pragma comment(lib, "winmm.lib")

namespace {

    // gaze dot colours on the result screen
    constexpr float LEFT_EYE_RGB[3] = { 0.1f, 1.0f, 1.0f };   // cyan
    constexpr float RIGHT_EYE_RGB[3] = { 1.0f, 0.1f, 1.0f };   // magenta

    // samples that tobii collected but then threw away are drawn faded
    constexpr float DISCARDED_SAMPLE_ALPHA = 0.35f;

    // debug mode (--debug): how many fake samples are generated per calibration point
    constexpr int FAKE_SAMPLES_PER_POINT = 30;

    // =============== head positioning guide
    // the guide is green while the midpoint between the eyes is within these distances of the centre of the track box
    constexpr float HEAD_TOLERANCE_XY = 0.20f; // left/right and up/down
    constexpr float HEAD_TOLERANCE_Z = 0.20f; // depth  
    // a tracker sample older than this counts as "no eyes" (head left the track box/stream stalled)
    constexpr double HEAD_SAMPLE_MAX_AGE_SEC = 0.3;
    // true: enter/A is ignored until the guide is green; back/B always returns to the start screen.
    constexpr bool REQUIRE_GOOD_POSITION = true;
    // confirm is ignored for this long after entering the stage (so a double press on the start screen can't skip it)
    constexpr double HEAD_CONFIRM_DELAY_SEC = 0.5;

    //************ debug mode: how fast the simulated head moves while a key is held (track box units per second)
    constexpr float DEBUG_HEAD_SPEED = 0.35f;

    // raw track box coordinates -> the participant's point of view
    // TBCS x points to the user's LEFT, so it is flipped: x = 0 is the participant's left, 1 their right
    // y (down) and z (towards the user: 0 = closest to the tracker, 1 = furthest) already read the way the guide wants them
    HeadGuideEye toGuideEye(const TrackBoxEye& e) {
        HeadGuideEye g;
        g.valid = e.valid;
        g.x = 1.0f - e.x;
        g.y = e.y;
        g.z = e.z;
        return g;
    }

    EyeSample toEyeSample(const TobiiResearchCalibrationEyeData& e) {
        EyeSample s;
        s.x = e.position_on_display_area.x;
        s.y = e.position_on_display_area.y;
        switch (e.validity) {
            case TOBII_RESEARCH_CALIBRATION_EYE_VALIDITY_VALID_AND_USED: 
            {
                s.validity = EyeValidity::ValidUsed;    
                break;
            }
            case TOBII_RESEARCH_CALIBRATION_EYE_VALIDITY_VALID_BUT_NOT_USED: 
            {
                s.validity = EyeValidity::ValidNotUsed;
                break;
            }
            default:                                                         
            {
                s.validity = EyeValidity::Invalid;
                break;
            }
        }
        return s;
    }

    const char* validityName(EyeValidity v) {
        switch (v) {
            case EyeValidity::ValidUsed:    
            {
                return "valid_used";
            }
            case EyeValidity::ValidNotUsed: 
            {
                return "valid_not_used";
            }
            {
            default:
                return "invalid";
            }
        }
    }

    // invalid samples can carry NaN coordinates; leave those cells empty
    void writeCoord(std::ostream& out, float v) {
        if (std::isfinite(v)) out << v;
    }

    // prints a display area in the same layout config.json expects, so it can be copied straight in
    void printDisplayArea(const char* label, const TobiiResearchDisplayArea& a) {
        auto point = [](const TobiiResearchPoint3D& p) {
            std::ostringstream s;
            s << std::fixed << std::setprecision(3) << "[" << p.x << ", " << p.y << ", " << p.z << "]";
            return s.str();
            };

        std::cout << "[DisplayArea] " << label << " (mm, tracker coordinate system):\n"
            << "  \"Display Area (mm)\": {\n"
            << "    \"Top Left\": " << point(a.top_left) << ",\n"
            << "    \"Top Right\": " << point(a.top_right) << ",\n"
            << "    \"Bottom Left\": " << point(a.bottom_left) << "\n"
            << "  }\n";
    }

} // namespace

/// <summary>
/// Handles the app's lifecycle
/// </summary>
App::~App() {
    stopGazeStream(); // the Tobii thread calls back into this object
    leaveCalibrationMode(); // never leave the tracker stuck in calibration mode (e.g. Esc mid-calibration)
    m_renderer.waitIdle(); // make sure GPU is done before window dies
    if (m_window) glfwDestroyWindow(m_window);
    glfwTerminate();
}


/// <summary>
/// Initialize the app.
/// </summary>
/// <param name="configPath">Location of config file. Defaults to 'config.json' in .exe location. </param>
/// <returns>True if app successsfully initialized. False otherwise. </returns>
bool App::init(const std::string& configPath) {

    if (!m_config.loadConfig(configPath)) return false;

    // *************** GLFW init and window creation **********************
    if (!glfwInit()) { Utils::FatalError("[App] GLFW init failed"); return false; }

    // *************** participant id **********************
    // asked before the (borderless, double wide) window exists, so the popup isn't hidden behind it
    std::string participantId;
    if (!ParticipantPrompt::ask(participantId)) {
        std::cerr << "[App] No participant ID entered, exiting.\n";
        return false;
    }
    m_config.participantId = participantId;
    std::cout << "[App] Participant ID: " << m_config.participantId << "\n";

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    m_displayInfo.pixelWidth = mode->width;
    m_displayInfo.pixelHeight = mode->height;


    m_window = glfwCreateWindow(
        m_displayInfo.pixelWidth * 2, // instead of dealing with two windows, use 1 window stretched to fit 2 monitors. ( this is so i dont have to deal with switching contexts all the time)
        m_displayInfo.pixelHeight,
        "Tobii Eye Tracker Calibration",
        nullptr,
        nullptr
    );
    glfwSetWindowUserPointer(m_window, this);
    glfwSetKeyCallback(m_window, keyCallback);
    glfwSetFramebufferSizeCallback(m_window, framebufferSizeCallback);
    glfwGetMonitorPhysicalSize(monitor, &m_displayInfo.physicalWidthMm, &m_displayInfo.physicalHeightMm); // returns sizes in mm
    buildCalibrationTargets(); // needs the physical size


    std::cerr << "[Config] Successfully loaded: " + configPath + "\n";
    std::cout << m_config.toString() << std::endl;

    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // disable cursor

    // init the renderer
    if (!m_renderer.init(m_window, m_displayInfo.pixelWidth, m_displayInfo.pixelHeight))
        return false;

    // upload the textures (instructions, fail, and complete screens.)
    try {
        loadInstructionsTextures();
    }
    catch (const std::exception& e) {
        Utils::FatalError(std::string("[App] Failed to load instruction textures: ") + e.what());
        return false;
    }

    // *************** find and connect to the eye tracker **********************

    if (m_debugNoTracker) {
        // debug mode: no tracker is touched at all, calibration data is synthetic
        std::cout << "[DEBUG] Debug mode: no eye tracker will be used, calibration data is synthetic.\n";
    }
    else {
        TobiiResearchEyeTrackers* eyetrackers = nullptr;
        TobiiResearchStatus findStatus = tobii_research_find_all_eyetrackers(&eyetrackers);

        if (findStatus != TOBII_RESEARCH_STATUS_OK || !eyetrackers || eyetrackers->count == 0) {
            Utils::FatalError("[App] No Tobii eye tracker found. (Run with --debug to test without one.)");
            return false;
        }

        m_tobiiHandle = eyetrackers->eyetrackers[0]; // handle stays valid after the list itself is freed
        tobii_research_free_eyetrackers(eyetrackers);

        if (!setUpDisplayArea()) return false;
    }

    setPhase(CalibrationStage::StartInstructions);

    return true;
}

/// <summary>
/// Tells the tracker where the screen is
/// 
/// - always prints the display area currently stored on the tracker
/// - if the config has a "Display Area (mm)" it is written to the tracker, then read back and printed. Applied every startup
/// - a failure to apply is fatal
/// </summary>
bool App::setUpDisplayArea() {
    TobiiResearchDisplayArea current{};
    if (tobii_research_get_display_area(m_tobiiHandle, &current) == TOBII_RESEARCH_STATUS_OK)
    {
        printDisplayArea("Stored on the tracker", current);
    }
    else
    {
        std::cerr << "[DisplayArea] Could not read the display area from the tracker.\n";
    }

    if (!m_config.displayArea) {
        std::cout << "[DisplayArea] No \"Display Area (mm)\" in the config: using the one stored on the tracker.\n";
        return true;
    }

    const DisplayAreaConfig& want = *m_config.displayArea;

    TobiiResearchDisplayArea area{};

    area.top_left.x = want.topLeft[0];
    area.top_left.y = want.topLeft[1];
    area.top_left.z = want.topLeft[2];
    area.top_right.x = want.topRight[0];
    area.top_right.y = want.topRight[1];
    area.top_right.z = want.topRight[2];
    area.bottom_left.x = want.bottomLeft[0];
    area.bottom_left.y = want.bottomLeft[1];
    area.bottom_left.z = want.bottomLeft[2];
    // bottom right, width and height are derived by the tracker

    const TobiiResearchStatus status = tobii_research_set_display_area(m_tobiiHandle, &area);
    if (status != TOBII_RESEARCH_STATUS_OK) {
        Utils::FatalError("[DisplayArea] The tracker rejected the display area from the config (status " +
            std::to_string(static_cast<int>(status)) + "). Not every tracker model supports setting it from software.");
        return false;
    }

    TobiiResearchDisplayArea applied{};
    if (tobii_research_get_display_area(m_tobiiHandle, &applied) == TOBII_RESEARCH_STATUS_OK)
    {
        printDisplayArea("Now on the tracker (from config)", applied);
    }
    else
    {
        std::cout << "[DisplayArea] Display area from the config was applied.\n";
    }

    return true;
}

/// <summary>
/// Main loop 
/// </summary>
void App::run() {
    while (!glfwWindowShouldClose(m_window) && m_phase != CalibrationStage::Done) { // auto close on complete
        glfwPollEvents();
        update();
        m_renderer.drawFrame(buildScene()); // draws the scene
    }
}


/// <summary>
/// Build scene passes on the app's state to the renderer.
/// The app handles all state changes, so for seperation of responsibility, 
/// we need to update a shared 'scene' that the app can update for the renderer to read.
/// </summary>
/// <returns>
/// FrameScene object
/// </returns>
FrameScene App::buildScene() const {
    FrameScene s;
    s.mode = m_phase;

    switch (m_phase) {
    case CalibrationStage::HeadPositioning:
        s.head = m_headGuide;
        break;

    case CalibrationStage::Calibration:
        // just the target we are currently collecting, as a white dot
        s.dots.push_back(m_targets[m_currentFixation]);
        break;

    case CalibrationStage::Complete:
        // every target (as crosses), with the gaze samples uploaded by computeCalibration() drawn around them
        s.crosses = m_targets;
        s.drawGazeDots = true;
        break;

    default:
        break;
    }
    return s;
}

/// <summary>
/// The main update loop.
/// </summary>
void App::update() {
    pollGamepad(); // collect button press events from the gamepad

    switch (m_phase) {
    case CalibrationStage::HeadPositioning:
        updateHeadPositioning();
        break;

    case CalibrationStage::Calibration:
        updateCalibration();
        break;

    case CalibrationStage::Calculating:
        // runs one frame AFTER the phase was entered, so the blank "calculating" frame has been presented
        // before the (blocking) tobii compute call
        computeCalibration();
        break;

    default:
        break; // start / complete / failure wait for input
    }
}

void App::setPhase(CalibrationStage next) {
    m_phase = next;
    m_phaseStart = glfwGetTime();
}


// ****************************** input ******************************

/// <summary>
/// A button / Enter.
/// </summary>
void App::onConfirm() {
    switch (m_phase) {
        case CalibrationStage::StartInstructions: 
        { beginHeadPositioning();
            break;
        }
        case CalibrationStage::HeadPositioning:
        {
            // only continue once the guide is green (and not within the first moments of the stage)
            if (glfwGetTime() - m_phaseStart >= HEAD_CONFIRM_DELAY_SEC && (!REQUIRE_GOOD_POSITION || m_headGuide.inPosition))
            {
                beginCalibration();
            }
            break;
        }
        case CalibrationStage::Failure:
        {
            returnToStart();
            break;
        }
        case CalibrationStage::Complete:
        {
            acceptCalibration(); break;
        }
        default: 
        {
            break; // ignored while calibrating / calculating
        }
    }
}

/// <summary>
/// B button / backspac; on the success screen: reject -> redo. On the head positioning screen: back to the start screen.
/// </summary>
void App::onBack() {
    if (m_phase == CalibrationStage::Complete) {
        returnToStart();
    }
    else if (m_phase == CalibrationStage::HeadPositioning) {
        stopGazeStream();
        setPhase(CalibrationStage::StartInstructions);
    }
}

/// <summary>
/// Polls the gamepad for button presses (edge triggered)
/// </summary>
void App::pollGamepad() {
    GLFWgamepadstate state;
    if (!glfwGetGamepadState(GLFW_JOYSTICK_1, &state)) return;

    const bool aPressed = state.buttons[GLFW_GAMEPAD_BUTTON_A] == GLFW_PRESS;
    const bool bPressed = state.buttons[GLFW_GAMEPAD_BUTTON_B] == GLFW_PRESS;

    if (aPressed && !m_prevGamepadA) onConfirm();
    if (bPressed && !m_prevGamepadB) onBack();

    m_prevGamepadA = aPressed;
    m_prevGamepadB = bPressed;
}


// ****************************** head positioning ******************************

/// <summary>
/// ubscribes to the tracker's user position guide stream (the position of each eye in track box coordinates)
/// The stream is stopped again before calibration mode is entered.
/// </summary>
bool App::startGazeStream() {
    if (m_gazeStreamActive) return true;

    const TobiiResearchStatus status =  tobii_research_subscribe_to_user_position_guide(m_tobiiHandle, &App::userPositionCallback, this);
    if (status != TOBII_RESEARCH_STATUS_OK) {
        std::cerr << "[HeadPositioning] subscribe_to_user_position_guide failed (status " << status << ")\n";
        return false;
    }
    m_gazeStreamActive = true;
    return true;
}

void App::stopGazeStream() {
    if (!m_gazeStreamActive) return;
    tobii_research_unsubscribe_from_user_position_guide(m_tobiiHandle, &App::userPositionCallback);
    m_gazeStreamActive = false;
}

void App::userPositionCallback(TobiiResearchUserPositionGuide* guide, void* userData) {
    if (guide && userData) static_cast<App*>(userData)->onUserPositionSample(*guide);
}

/// <summary>
/// runs on the Tobii thread: keeps only the newest eye positions (*********) 
/// valid when the tracker says its position is valid and the numbers are finite
/// </summary>
void App::onUserPositionSample(const TobiiResearchUserPositionGuide& guide) {
    auto convert = [](const TobiiResearchEyeUserPositionGuide& e) {
        const TobiiResearchNormalizedPoint3D& p = e.user_position;

        TrackBoxEye t;
        t.valid = e.validity == TOBII_RESEARCH_VALIDITY_VALID &&   std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
        if (t.valid) {
            t.x = p.x;
            t.y = p.y;
            t.z = p.z;
        }

        return t;
    };

    TrackBoxSample s;
    s.left = convert(guide.left_eye);
    s.right = convert(guide.right_eye);
    s.stamp = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(m_trackBoxMutex);
    m_trackBoxSample = s;
}

/// <summary>
/// StartInstructions -> HeadPositioning.
/// Starts streaming the eye positions inside the track box (real tracker only) and shows the guide.
/// Debug mode: no tracker; head starts exactly in the ideal position 
/// </summary>
void App::beginHeadPositioning() {
    m_headGuide = HeadGuide{};
    m_debugHeadDx = m_debugHeadDy = m_debugHeadDz = 0.0f;
    {
        std::lock_guard<std::mutex> lock(m_trackBoxMutex);
        m_trackBoxSample = TrackBoxSample{}; // forget eyes from an earlier visit to this stage
    }

    if (!m_debugNoTracker && !startGazeStream()) {
        std::cerr << "[HeadPositioning] Could not start the gaze data stream.\n";
        setPhase(CalibrationStage::Failure);
        return;
    }

    m_lastHeadUpdate = glfwGetTime();
    setPhase(CalibrationStage::HeadPositioning);
}

/// <summary>
/// - runs frames while in the HeadPositioning phase
/// - rebuilds m_headGuide from the latest eye positions
/// - Debug mode simulates a head that sits in the ideal position; the arrow keys, PageUp / PageDown and Home to test w/out tracker
/// </summary>
void App::updateHeadPositioning() {
    const double now = glfwGetTime();
    const float dt = static_cast<float>(now - m_lastHeadUpdate);
    m_lastHeadUpdate = now;

    TrackBoxEye left, right;

    if (m_debugNoTracker) {
        auto held = [&](int key) { return glfwGetKey(m_window, key) == GLFW_PRESS; };
        const float step = DEBUG_HEAD_SPEED * dt;
        if (held(GLFW_KEY_LEFT))      m_debugHeadDx -= step;
        if (held(GLFW_KEY_RIGHT))     m_debugHeadDx += step;
        if (held(GLFW_KEY_UP))        m_debugHeadDy -= step;
        if (held(GLFW_KEY_DOWN))      m_debugHeadDy += step;
        if (held(GLFW_KEY_PAGE_UP))   m_debugHeadDz -= step; // closer to the screen
        if (held(GLFW_KEY_PAGE_DOWN)) m_debugHeadDz += step; // further away
        if (held(GLFW_KEY_HOME))      m_debugHeadDx = m_debugHeadDy = m_debugHeadDz = 0.0f;
        m_debugHeadDx = std::clamp(m_debugHeadDx, -0.6f, 0.6f);
        m_debugHeadDy = std::clamp(m_debugHeadDy, -0.6f, 0.6f);
        m_debugHeadDz = std::clamp(m_debugHeadDz, -0.6f, 0.6f);

        // build it the way the tracker would report it (raw track box coordinates) 
        const float midX = 0.5f - m_debugHeadDx;
        left = { true, midX + HEAD_GUIDE_IDEAL_EYE_HALF_SEPARATION, 0.5f + m_debugHeadDy, 0.5f + m_debugHeadDz };
        right = { true, midX - HEAD_GUIDE_IDEAL_EYE_HALF_SEPARATION, 0.5f + m_debugHeadDy, 0.5f + m_debugHeadDz };
    }
    else {
        TrackBoxSample sample;
        {
            std::lock_guard<std::mutex> lock(m_trackBoxMutex);
            sample = m_trackBoxSample;
        }
        const double age = std::chrono::duration<double>(std::chrono::steady_clock::now() - sample.stamp).count();
        if (age <= HEAD_SAMPLE_MAX_AGE_SEC) {
            left = sample.left;
            right = sample.right;
        }
    }

    HeadGuide g;
    g.left = toGuideEye(left);
    g.right = toGuideEye(right);
    g.depthTolerance = HEAD_TOLERANCE_Z;

    if (g.left.valid && g.right.valid) {
        const float midX = 0.5f * (g.left.x + g.right.x);
        const float midY = 0.5f * (g.left.y + g.right.y);
        const float midZ = 0.5f * (g.left.z + g.right.z);
        g.inPosition = std::fabs(midX - 0.5f) <= HEAD_TOLERANCE_XY &&
            std::fabs(midY - 0.5f) <= HEAD_TOLERANCE_XY &&
            std::fabs(midZ - 0.5f) <= HEAD_TOLERANCE_Z;
    }
    m_headGuide = g;
}



// ****************************** calibration ******************************

/// <summary>
/// Enters calibration mode.
/// Debug mode:there is no tracker, so this just resets state and starts the first point.
/// </summary>
void App::beginCalibration() {
    stopGazeStream(); // the head positioning stream is not needed any more, and shouldn't run during calibration

    if (!m_debugNoTracker) {
        TobiiResearchStatus status = tobii_research_screen_based_calibration_enter_calibration_mode(m_tobiiHandle);
        if (status != TOBII_RESEARCH_STATUS_OK) {
            // a previous run may have left the tracker in calibration mode: leave it and try once more
            tobii_research_screen_based_calibration_leave_calibration_mode(m_tobiiHandle);
            status = tobii_research_screen_based_calibration_enter_calibration_mode(m_tobiiHandle);
        }
        if (status != TOBII_RESEARCH_STATUS_OK) {
            std::cerr << "[Calibration] Could not enter calibration mode (status " << status << ")\n";
            setPhase(CalibrationStage::Failure);
            return;
        }

        m_inCalibrationMode = true;
    }

    m_results.clear();
    m_renderer.clearGazeDots();
    m_currentFixation = 0;
    setPhase(CalibrationStage::Calibration); // starts the timer for the first point
}

/// <summary>
/// Runs every frame while in the Calibration phase
/// </summary>
void App::updateCalibration() {
    const double elapsed = glfwGetTime() - m_phaseStart;
    if (elapsed * 1000.0 < fixationTimeMs) return; // user is still settling on the point

    collectCurrentPoint();

    if (m_currentFixation + 1 < static_cast<int>(m_targets.size())) {
        ++m_currentFixation;
        m_phaseStart = glfwGetTime(); // settle timer for the next point (collect above blocked for a while)
    }
    else {
        setPhase(CalibrationStage::Calculating);
    }
}

/// <summary>
/// Collects tobii data for the current fixation point
/// Blocking (the dot stays on screen, since the last presented frame keeps being displayed).
/// - Debug mode: samples are randomized
/// </summary>
void App::collectCurrentPoint() {
    if (m_debugNoTracker) return; // nothing to collect; samples are randomized in computeFakeCalibration()

    const NormalizedPoint2D& p = m_targets[m_currentFixation];

    if (tobii_research_screen_based_calibration_collect_data(m_tobiiHandle, p.x, p.y) != TOBII_RESEARCH_STATUS_OK) {
        {
            // try again if it didnt go well the first time 
            if (tobii_research_screen_based_calibration_collect_data(m_tobiiHandle, p.x, p.y) != TOBII_RESEARCH_STATUS_OK)
            {
                std::cerr << "[Calibration] collect_data failed twice at (" << p.x << ", " << p.y << ")\n";
            }
        }
    }
}

/// <summary>
/// Calculating -> Complete | Failure.
/// compute_and_apply, copy the result out of the tobii structs, leave calibration mode.
/// </summary>
void App::computeCalibration() {
    if (m_debugNoTracker) { computeFakeCalibration(); return; }

    TobiiResearchCalibrationResult* result = nullptr;
    const TobiiResearchStatus status = tobii_research_screen_based_calibration_compute_and_apply(m_tobiiHandle, &result);

    const bool success = status == TOBII_RESEARCH_STATUS_OK && result != nullptr &&  result->status == TOBII_RESEARCH_CALIBRATION_SUCCESS;

    m_results.clear();
    if (success) {
        std::cout << "[Calibration] Compute and apply returned " << status << " and collected at " << result->calibration_point_count << " points.\n";

        for (size_t i = 0; i < result->calibration_point_count; ++i) {
            const TobiiResearchCalibrationPoint& pt = result->calibration_points[i];

            CalibrationPointData data;
            data.target.x = pt.position_on_display_area.x;
            data.target.y = pt.position_on_display_area.y;
            data.samples.reserve(pt.calibration_sample_count);

            for (size_t j = 0; j < pt.calibration_sample_count; ++j) {
                CalibrationSample s;
                s.left = toEyeSample(pt.calibration_samples[j].left_eye);
                s.right = toEyeSample(pt.calibration_samples[j].right_eye);
                data.samples.push_back(s);
            }
            m_results.push_back(std::move(data));
        }
    }
    else {
        std::cerr << "[Calibration] Calibration failed! (status " << status << ")\n";
    }

    /* Free calibration result when done using it */
    if (result) tobii_research_free_screen_based_calibration_result(result);

    /* The calibration is done. Leave calibration mode. */
    leaveCalibrationMode();

    if (success) {
        m_renderer.setGazeDots(buildGazeDots());
        setPhase(CalibrationStage::Complete);
    }
    else {
        m_results.clear();
        setPhase(CalibrationStage::Failure);
    }
}

/// <summary>
/// Debug only: builds fake calibration results (gaussian scatter around each target,
/// ~10% valid-but-discarded, ~5% invalid with NaN coordinates) to cover all render paths
/// then goes through the same Complete path as a real calibration.
/// **** hold Shift while the "calculating" screen is up to force the Failure screen instead
/// </summary>
void App::computeFakeCalibration() {
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> noise(0.0f, 0.01f);
    std::uniform_real_distribution<float> roll(0.0f, 1.0f);

    auto fakeEye = [&](const NormalizedPoint2D& t) {
        EyeSample e;
        const float r = roll(rng);
        if (r < 0.05f) {
            e.x = e.y = std::numeric_limits<float>::quiet_NaN();
            e.validity = EyeValidity::Invalid;
        }
        else {
            e.x = t.x + noise(rng);
            e.y = t.y + noise(rng);
            e.validity = (r < 0.15f) ? EyeValidity::ValidNotUsed : EyeValidity::ValidUsed;
        }
        return e;
        };

    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // so the "calculating" screen is visible

    // hold shift to test the Failure screen
    if (GetAsyncKeyState(VK_SHIFT) & 0x8000) {
        std::cerr << "[DEBUG] Shift held: forcing calibration failure.\n";
        m_results.clear();
        setPhase(CalibrationStage::Failure);
        return;
    }

    m_results.clear();
    for (const NormalizedPoint2D& t : m_targets) {
        CalibrationPointData data;
        data.target = t;
        data.samples.reserve(FAKE_SAMPLES_PER_POINT);
        for (int j = 0; j < FAKE_SAMPLES_PER_POINT; ++j) {
            CalibrationSample s;
            s.left = fakeEye(t);
            s.right = fakeEye(t);
            data.samples.push_back(s);
        }
        m_results.push_back(std::move(data));
    }

    std::cout << "[DEBUG] Generated fake calibration with " << m_results.size() << " points.\n";
    m_renderer.setGazeDots(buildGazeDots());
    setPhase(CalibrationStage::Complete);
}

void App::leaveCalibrationMode() {
    if (!m_inCalibrationMode) return;
    tobii_research_screen_based_calibration_leave_calibration_mode(m_tobiiHandle);
    m_inCalibrationMode = false;
}

/// <summary>
/// Failure --confirm--> start, or Complete --back--> start.
/// </summary>
void App::returnToStart() {
    m_renderer.clearGazeDots();
    m_results.clear();
    m_currentFixation = 0;
    setPhase(CalibrationStage::StartInstructions);
}

/// <summary>
/// Complete --confirm--> Done. The calibration was already applied to the tracker by compute_and_apply,
/// so accepting just saves the collected points. If the csv can't be written we stay on the
/// result screen so it can be retried
/// </summary>
void App::acceptCalibration() {
    if (!writeCalibrationCsv()) {
        std::cerr << "[Calibration] Could not write the calibration csv; staying on the result screen.\n";
        return;
    }
    setPhase(CalibrationStage::Done);
}

/// <summary>
/// Maps the fixation points from the calibration region to the screen.
/// 
/// The calibration region is a rectangle of (calibration width x height) mm, centred on the monitor.
/// m_fixationPoints are relative to that rectangle; tobii (and the renderer) want coordinates relative
/// to the whole display, so each point is pulled towards the screen centre by the region/screen ratio: 
/// screen = 0.5 + (point - 0.5) * (regionMm / screenMm)
/// 
/// - an axis without a configured size (or with a size larger than the screen) uses the full screen.
/// </summary>
void App::buildCalibrationTargets() {
    auto regionFraction = [](double regionMm, int screenMm, const char* axis) -> float {
        if (regionMm <= 0.0) return 1.0f; // not configured: whole screen

        if (screenMm <= 0) {
            std::cerr << "[Calibration] Warning: the monitor did not report its physical " << axis   << ", so the calibration " << axis << " can't be applied. Using the full screen.\n";
            return 1.0f;
        }
        if (regionMm > screenMm) {
            std::cerr << "[Calibration] Warning: calibration " << axis << " (" << regionMm  << " mm) is larger than the screen (" << screenMm << " mm). Using the full screen.\n";
            return 1.0f;
        }
        return static_cast<float>(regionMm / screenMm);
        };

    const float fx = regionFraction(m_config.calibrationWidthMm, m_displayInfo.physicalWidthMm, "width");
    const float fy = regionFraction(m_config.calibrationHeightMm, m_displayInfo.physicalHeightMm, "height");

    m_targets.clear();
    for (const NormalizedPoint2D& p : m_fixationPoints) {
        NormalizedPoint2D t;
        t.x = 0.5f + (p.x - 0.5f) * fx;
        t.y = 0.5f + (p.y - 0.5f) * fy;
        m_targets.push_back(t);
    }

    std::cout << "[Calibration] Screen " << m_displayInfo.physicalWidthMm << " x " << m_displayInfo.physicalHeightMm  << " mm, calibration region covers " << fx * 100.0f << "% x " << fy * 100.0f << "% of it.\n";
}

/// <summary>
/// Turns the stored calibration samples into dots for the renderer: left eye cyan, right eye magenta.
/// Invalid samples are skipped, samples tobii discarded are drawn faded.
/// </summary>
std::vector<GazeDot> App::buildGazeDots() const {
    std::vector<GazeDot> dots;

    auto add = [&](const EyeSample& e, const float* rgb) {
        if (e.validity == EyeValidity::Invalid) return;
        if (!std::isfinite(e.x) || !std::isfinite(e.y)) return;

        GazeDot d;
        d.x = e.x;
        d.y = e.y;
        d.r = rgb[0];
        d.g = rgb[1];
        d.b = rgb[2];
        d.a = (e.validity == EyeValidity::ValidUsed) ? 1.0f : DISCARDED_SAMPLE_ALPHA;
        dots.push_back(d);
        };

    for (const CalibrationPointData& pt : m_results) {
        for (const CalibrationSample& s : pt.samples) {
            add(s.left, LEFT_EYE_RGB);
            add(s.right, RIGHT_EYE_RGB);
        }
    }
    return dots;
}

/// <summary>
/// - Writes one row per tobii calibration sample to <output directory>/calibration_<participantId>_<yyyymmdd_hhmmss>.csv
/// - prefixed with DEBUG_ in debug mode
/// - Coordinates are normalized display-area coordinates (the space tobii reports them in).
/// </summary>
bool App::writeCalibrationCsv() const {
    namespace fs = std::filesystem;
    try {
        const std::time_t now = std::time(nullptr);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &now);
#else
        localtime_r(&now, &tm);
#endif
        std::ostringstream stamp;
        stamp << std::put_time(&tm, "%Y%m%d_%H%M%S");

        const std::string participant = m_config.participantId; // already sanitized

        const fs::path dir = m_config.outputDirectory; // from config.json, created when the config was loaded
        fs::create_directories(dir);
        const std::string prefix = m_debugNoTracker ? "DEBUG_" : "";
        const fs::path file = dir / (prefix + "calibration_" + participant + "_" + stamp.str() + ".csv");

        std::ofstream out(file);
        if (!out) return false;

        out << std::setprecision(9);
        out << "participant_id,session_timestamp,point_index,target_x,target_y,sample_index,left_x,left_y,left_validity,right_x,right_y,right_validity\n";

        for (size_t p = 0; p < m_results.size(); ++p) {
            const CalibrationPointData& pt = m_results[p];
            for (size_t s = 0; s < pt.samples.size(); ++s) {
                const CalibrationSample& sample = pt.samples[s];

                out << participant << ',' << stamp.str() << ',' << p << ',' << pt.target.x << ',' << pt.target.y << ',' << s << ',';
                writeCoord(out, sample.left.x);  out << ',';
                writeCoord(out, sample.left.y);  out << ',' << validityName(sample.left.validity) << ',';
                writeCoord(out, sample.right.x); out << ',';
                writeCoord(out, sample.right.y); out << ',' << validityName(sample.right.validity) << '\n';
            }
        }

        out.flush();
        if (!out) return false; 
        std::cout << "[Calibration] Wrote " << file.string() << "\n";
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[Calibration] CSV write failed: " << e.what() << "\n";
        return false;
    }
}

// loading textures
// hardcoded to load textures for the instruction screens. these are loaded in once per program's lifecycle

/// <summary>
/// Loads the instruction screens (start, calibration failed, calibration complete), one image per monitor.
/// These are loaded once per program lifecycle, and are kept as unchanging textures throughout.
/// Images go through Renderer::decodeImageForUpload, which does the left/right mirroring for the wheatstone setup.
/// </summary>
void App::loadInstructionsTextures()
{
    const auto assetsDir = Utils::getExecutableDirectory() / "assets" / "instructions";

    m_renderer.decodeAndUploadTexture(TEX_START, (assetsDir / "calibration_start.ppm").string());

    m_renderer.decodeAndUploadTexture(TEX_CALCULATING, (assetsDir / "calculating_screen.ppm").string());

    m_renderer.decodeAndUploadTexture(TEX_SAVING, (assetsDir / "saving_screen.ppm").string());

    m_renderer.decodeAndUploadTexture(TEX_FAIL, (assetsDir / "calibration_failed.ppm").string());

    m_renderer.decodeAndUploadTexture(TEX_SUCCESS, (assetsDir / "calibration_success.ppm").string());
}


// GLFW callbacks

void App::keyCallback(GLFWwindow* window, int key, int /*scancode*/,
    int action, int /*mods*/)
{
    if (action != GLFW_PRESS) return;
    App* app = static_cast<App*>(glfwGetWindowUserPointer(window));

    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(window, true);
        return;
    }
    if (key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER)
        app->onConfirm();
    else if (key == GLFW_KEY_BACKSPACE)
        app->onBack();
}