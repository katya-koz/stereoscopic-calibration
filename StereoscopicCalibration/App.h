#pragma once

#include "render.h"
#include "rendertypes.h"
#include "config.h"
#include "utils.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>
#include "tobii_research.h"
#include "tobii_research_eyetracker.h"
#include "tobii_research_calibration.h"
#include "tobii_research_streams.h"
#include "DisplayInfo.h"
#include "CalibrationStage.h"
#include "CalibrationTypes.h"

 
struct TrackBoxEye {
    bool  valid = false;
    float x = 0.5f;
    float y = 0.5f;
    float z = 0.5f;
};

// latest gaze origin sample: written by the Tobii callback thread and read by the main thread
struct TrackBoxSample {
    TrackBoxEye left;
    TrackBoxEye right;
    std::chrono::steady_clock::time_point stamp{}; // when it arrived; default (epoch) = never
};

class App {
public:
    App() = default;
    ~App();

    App(const App&) = delete;
    App& operator=(const App&) = delete;

    bool init(const std::string& configPath);
    void run();
    void setDebugNoTracker(bool v) { m_debugNoTracker = v; }

    void onUserPositionSample(const TobiiResearchUserPositionGuide& guide); // Tobii thread
    static void userPositionCallback(TobiiResearchUserPositionGuide* guide, void* userData); // Tobii thread


private:
   
    // ---- tracker setup
    bool setUpDisplayArea(); // print the tracker's display area; apply the one from the config if there is one
    bool m_debugNoTracker = false; // run without a tracker; calibration data is synthetic
    void computeFakeCalibration();
    // ---- main loop pieces
    void update();
    void pollGamepad();

    // translate current phase into a scene description so that the renderer can draw
    FrameScene buildScene() const;

    // ---- state machine 
    //   StartInstructions --confirm--> HeadPositioning --confirm (only while green)--> Calibration
    //   HeadPositioning --back--> StartInstructions
    //   Calibration --(all points collected)--> Calculating
    //   Calculating --> Complete | Failure
    //   Failure --confirm--> StartInstructions
    //   Complete --back--> StartInstructions
    //   Complete --confirm--> (write csv) --> Done 
    // confirm = A button / Enter, back = B button / Backspace
    void setPhase(CalibrationStage next); // changes phase and restarts the phase timer
    void onConfirm();
    void onBack(); 
    // ---- head positioning (between the start screen and the calibration)
    void beginHeadPositioning(); // start the gaze stream, show the guide
    void updateHeadPositioning(); // per frame: rebuild m_headGuide from the latest eye positions 
    bool startGazeStream(); // subscribe to the tracker's gaze data (gaze origin in track box coordinates)
    void stopGazeStream(); // safe to call at any time
    // ---- calibration steps
    void beginCalibration(); // enter Tobii calibration mode and show the first point
    void updateCalibration(); // per frame timing (not sleep): wait for the fixation time, then collect and advance
    void collectCurrentPoint(); // tobii collect_data at the current fixation point ********(blocking)
    void computeCalibration(); // tobii compute_and_apply -> Complete or Failure
    void leaveCalibrationMode(); // safe to call at any time
    void returnToStart();
    void acceptCalibration(); // write csv -> Done

    // converts m_fixationPoints (relative to the calibration region) into m_targets (relative to the screen)
    void buildCalibrationTargets();

    std::vector<GazeDot> buildGazeDots() const;
    bool writeCalibrationCsv() const;

    void loadInstructionsTextures();

    // glfw callbacks
    static void keyCallback(GLFWwindow*, int, int, int, int);
    static void framebufferSizeCallback(GLFWwindow*, int, int) {} // unused

    int fixationTimeMs = 3000;

    // show the fixation points in this order
    // relative to the CALIBRATION REGION (config: calibration width / height), where (0,0) is the top-left
    // of that region and (1,1) the bottom-right.
    std::vector<NormalizedPoint2D> m_fixationPoints =
    {
        { 0.5f, 0.5f },
        { 0.1f, 0.1f },
        { 0.1f, 0.9f },
        { 0.9f, 0.1f },
        { 0.9f, 0.9f },
    };

    // the same points relative to the whole screen (tobii display area coordinates).
    // This is what gets drawn and what is passed to tobii
    std::vector<NormalizedPoint2D> m_targets;
    int m_currentFixation = 0;

    // what tobii collected, filled in by computeCalibration() on success
    std::vector<CalibrationPointData> m_results;
    bool m_inCalibrationMode = false;

    // head positioning
    std::mutex m_trackBoxMutex; // guards m_trackBoxSample (written on the Tobii thread)
    TrackBoxSample m_trackBoxSample;
    bool m_gazeStreamActive = false;
    HeadGuide m_headGuide; // what the renderer draws; rebuilt every frame while in HeadPositioning
    double m_lastHeadUpdate = 0.0;
    // debug mode only: the simulated head's offset from the ideal position (arrow keys / PageUp / PageDown / Home)
    float m_debugHeadDx = 0.0f; // + = participant's right
    float m_debugHeadDy = 0.0f; // + = down
    float m_debugHeadDz = 0.0f; // + = further from the screen

    DisplayInfo m_displayInfo;

    TobiiResearchEyeTracker* m_tobiiHandle = nullptr; // set once during init() from tobii_research_find_all_eyetrackers

    GLFWwindow* m_window = nullptr;
    Renderer m_renderer; // backend renderer
    Config m_config;
    double m_phaseStart = 0.0;

    // for the state machine
    CalibrationStage m_phase = CalibrationStage::StartInstructions;

    bool m_prevGamepadA = false;
    bool m_prevGamepadB = false;
};