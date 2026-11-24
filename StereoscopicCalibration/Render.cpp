#include "render.h"
#include "shaders.h"

#include <opencv2/opencv.hpp>
#include "Utils.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <thread>



// validation layers for debug builds
#ifdef NDEBUG
static constexpr bool ENABLE_VALIDATION = false;
#else
static constexpr bool ENABLE_VALIDATION = true;
#endif

// for querying monitor stats
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#pragma comment(lib, "dxgi.lib")
#endif

static const std::vector<const char*> VALIDATION_LAYERS = {
    "VK_LAYER_KHRONOS_validation"
};

// ensure that the GPU supports swapchains and HDR metadata
static const std::vector<const char*> DEVICE_EXTENSIONS = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

// fixationPoint geometry (61x61 px, 5px thick)
static constexpr float CROSSHAIR_SIZE_PX = 61.0f;
static constexpr float CROSSHAIR_THICKNESS_PX = 5.0f;

static constexpr float CROSSHAIR_HALF_THICKNESS_RATIO = CROSSHAIR_THICKNESS_PX / CROSSHAIR_SIZE_PX;

// these vertices define the shape of the cross hair
static const float CROSSHAIR_VERTS[] = {
    // horizontal
    -1.0f, -CROSSHAIR_HALF_THICKNESS_RATIO,
     1.0f, -CROSSHAIR_HALF_THICKNESS_RATIO,
     1.0f,  CROSSHAIR_HALF_THICKNESS_RATIO,

    -1.0f, -CROSSHAIR_HALF_THICKNESS_RATIO,
     1.0f,  CROSSHAIR_HALF_THICKNESS_RATIO,
    -1.0f,  CROSSHAIR_HALF_THICKNESS_RATIO,

    // vertical
    -CROSSHAIR_HALF_THICKNESS_RATIO, -1.0f,
     CROSSHAIR_HALF_THICKNESS_RATIO, -1.0f,
     CROSSHAIR_HALF_THICKNESS_RATIO,  1.0f,

    -CROSSHAIR_HALF_THICKNESS_RATIO, -1.0f,
     CROSSHAIR_HALF_THICKNESS_RATIO,  1.0f,
    -CROSSHAIR_HALF_THICKNESS_RATIO,  1.0f,
};

// gaze dot geometry (calibration result screen)
static constexpr float GAZE_DOT_RADIUS_PX = 5.0f;

// calibration target: a white dot (the crosses are only drawn on the Complete screen)
static constexpr float CALIBRATION_DOT_DIAMETER_PX = 30.0f;

// ---- head positioning guide -------------------------------------------------------------------
// Sizes are fractions of the monitor HEIGHT, so the guide looks the same at any resolution.
// The oval is the "head", the two rings are where the eyes should end up, the filled circles are the
// participant's actual eyes (they grow when the participant is too close and shrink when too far),
// and the bar underneath is a depth gauge (marker in the middle = ideal distance).
static constexpr float HEAD_OVAL_RX = 0.24f;  // oval half width
static constexpr float HEAD_OVAL_RY = 0.33f;  // oval half height
static constexpr float HEAD_OVAL_OUTLINE_PX = 6.0f;
static constexpr float HEAD_EYE_LINE_OFFSET = -0.03f; // eye level relative to the oval centre (negative = above)
static constexpr float HEAD_EYE_RING_RADIUS = 0.04f; // target ring radius
static constexpr float HEAD_EYE_RING_OUTLINE_PX = 4.0f;
static constexpr float HEAD_EYE_DOT_SCALE = 0.8f; // eye circle radius at ideal depth, relative to the ring
static constexpr float HEAD_DEPTH_GAIN = 1.2f; // how strongly depth changes the eye circle size
static constexpr float HEAD_RANGE_X = 1.2f; // track box width on screen (the whole box spans this many monitor heights)
static constexpr float HEAD_RANGE_Y = 0.9f; // track box height on screen
static constexpr float HEAD_GAUGE_Y = 0.41f; // depth gauge: distance below the screen centre
static constexpr float HEAD_GAUGE_HALF_WIDTH = 0.25f;

// The participant looks at the monitors through the stereoscope mirrors, so what they see is the horizontal
// mirror image of what is on the physical monitor (that is why image textures are flipped in decodeImageForUpload).
// The guide is laid out the way the participant should see it (move right -> the eyes move right) and then
// mirrored here
//

// linear colours (the swapchain is scRGB-linear or sRGB, so these are kept moderate for a dark room)
static constexpr float GUIDE_GOOD_RGB[3] = { 0.02f, 0.55f, 0.08f }; // green
static constexpr float GUIDE_BAD_RGB[3] = { 0.65f, 0.02f, 0.02f }; // red
static constexpr float GUIDE_NEUTRAL_RGB[3] = { 0.35f, 0.35f, 0.35f }; // grey

/// <summary>
/// Renderer lifecycle. Release all memory upon app termination.
/// Disassembles objects in the correct order ( delete children before parents, reverse creation order)
/// </summary>
Renderer::~Renderer() {
    if (m_device) vkDeviceWaitIdle(m_device); // ensure the GPU is not using any resources

    for (auto& tex : m_textures) destroyTexture(tex);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (m_imageAvailable[i]) vkDestroySemaphore(m_device, m_imageAvailable[i], nullptr);
        if (m_renderFinished[i]) vkDestroySemaphore(m_device, m_renderFinished[i], nullptr);
        if (m_inFlightFences[i]) vkDestroyFence(m_device, m_inFlightFences[i], nullptr);
    }

    if (m_fixationPointVB) vkDestroyBuffer(m_device, m_fixationPointVB, nullptr);
    if (m_fixationPointVBMem) vkFreeMemory(m_device, m_fixationPointVBMem, nullptr);

    if (m_gazeDotMapped) vkUnmapMemory(m_device, m_gazeDotVBMem);
    if (m_gazeDotVB) vkDestroyBuffer(m_device, m_gazeDotVB, nullptr);
    if (m_gazeDotVBMem) vkFreeMemory(m_device, m_gazeDotVBMem, nullptr);

    if (m_descPool) vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
    if (m_descSetLayout) vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);

    if (m_pipeline) vkDestroyPipeline(m_device, m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    if (m_fixationPointPipeline) vkDestroyPipeline(m_device, m_fixationPointPipeline, nullptr);
    if (m_fixationPointLayout) vkDestroyPipelineLayout(m_device, m_fixationPointLayout, nullptr);
    if (m_gazeDotPipeline) vkDestroyPipeline(m_device, m_gazeDotPipeline, nullptr);
    if (m_gazeDotLayout) vkDestroyPipelineLayout(m_device, m_gazeDotLayout, nullptr);
    if (m_shapePipeline) vkDestroyPipeline(m_device, m_shapePipeline, nullptr);
    if (m_shapeLayout) vkDestroyPipelineLayout(m_device, m_shapeLayout, nullptr);

    for (auto fb : m_framebuffers) vkDestroyFramebuffer(m_device, fb, nullptr);
    for (auto iv : m_swapImageViews) vkDestroyImageView(m_device, iv, nullptr);

    if (m_renderPass) vkDestroyRenderPass(m_device, m_renderPass, nullptr);
    if (m_swapchain) vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);

    if (m_commandPool) vkDestroyCommandPool(m_device, m_commandPool, nullptr);
    if (m_device) vkDestroyDevice(m_device, nullptr);

    if (ENABLE_VALIDATION && m_debugMessenger) {
        auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT");
        if (fn) fn(m_instance, m_debugMessenger, nullptr);
    }

    if (m_surface)  vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    if (m_instance) vkDestroyInstance(m_instance, nullptr);
}
/// <summary>
/// Creates every Vulkan object the renderer needs.
/// </summary>
/// <param name="window"> GLFW window created by App </param>
/// <param name="monitorWidth"> Width of monitors, (they are asssumed to be the same), passed down by App.</param>
/// <param name="monitorHeight"> Height of monitors, (they are asssumed to be the same), passed down by App.</param>
/// <param name="displayMode"> 0 = SDR, anything else = HDR </param>
/// <returns> True if successful init, false otherwise. </returns>
bool Renderer::init(GLFWwindow* window, int monitorWidth, int monitorHeight, int displayMode) {
    m_monitorWidth = monitorWidth;
    m_monitorHeight = monitorHeight;


    createInstance(); // root vulkan object
    if (ENABLE_VALIDATION) setupDebugMessenger(); // debug callbacks
    createSurface(window); // bind vulkan to the GLFW window
    pickPhysicalDevice(); // select GPU
    createLogicalDevice(); // create handles to the GPU
    createSwapChain(window); // creates frame buffer images to present to screen
    createImageViews(); // views into those swap images
    createRenderPass(); // describes render target layout transitions
    createDescriptorSetLayout(); // how the shaders bind to the textures
    createCommandPool(); // allocates GPU command buffers
    allocateDescriptorPool(); // allocates description sets

    createGraphicsPipeline(); // pipeline 1: textured quads (instruction / fail / complete screens)
    createFixationPointPipeline(); // pipeline 2: fixation cross
    createGazeDotPipeline(); // pipeline 3: instanced gaze dots
    createShapePipeline(); // pipeline 4: white calibration dots + head positioning guide

    createFramebuffers(); // wrap image views into frame buffer objects
    createFixationPointVertexBuffer(); // uploads cross geometry to the GPU
    createGazeDotBuffer(); // per-instance buffer for the gaze dots
    createCommandBuffers(); // creates command buffers,one per frame in flight
    createSyncObjects(); // create semaphores and fences for syncing
    return true;
}


void Renderer::waitIdle() {
    if (m_device) vkDeviceWaitIdle(m_device);
}


/// <summary>
/// Called every frame. Executes the frame loop.
/// </summary>
/// <param name="scene"> Pass through the scene to draw (current images, state, isFlicker, etc) </param>
void Renderer::drawFrame(const FrameScene& scene) {
    // wait until the GPU is done with the previous use of this frame slot
    // frames in flight is 2 (by default), so m_currentFrame cycles 0 -> 1 -> 0 -> 1... allows CPU to prepare N+1 while GPU renders N
    //( fences let the CPU know when the GPU has finished its work) CPU -> GPU syncronization
    vkWaitForFences(m_device, 1, &m_inFlightFences[m_currentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    // asks the swapchain which image should be rendered next. 
    VkResult res = vkAcquireNextImageKHR(
        m_device,
        m_swapchain,
        UINT64_MAX,
        m_imageAvailable[m_currentFrame],
        VK_NULL_HANDLE,
        &imageIndex // which image in swapchain to render
    );

    //if the window was resized, return
    if (res == VK_ERROR_OUT_OF_DATE_KHR) return;
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) throw std::runtime_error("vkAcquireNextImageKHR failed");

    // reset the fence
    vkResetFences(m_device, 1, &m_inFlightFences[m_currentFrame]);
    vkResetCommandBuffer(m_commandBuffers[m_currentFrame], 0);

    // re-record the frame's command buffer from scratch based on the current frame scene 
    recordCommandBuffer(m_commandBuffers[m_currentFrame], imageIndex, scene);

    // wait for imageAvailable semaphore, and signal renderFinished when done
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &m_imageAvailable[m_currentFrame];
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &m_commandBuffers[m_currentFrame];
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &m_renderFinished[m_currentFrame];

    // CPU submits work to the GPU
    if (vkQueueSubmit(m_graphicsQueue, 1, &si,
        m_inFlightFences[m_currentFrame]) != VK_SUCCESS)
        throw std::runtime_error("vkQueueSubmit failed");

    // wait for renderfinished, then tell the swapchain to present the next image
    VkPresentInfoKHR pi{};
    pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &m_renderFinished[m_currentFrame];
    pi.swapchainCount = 1;
    pi.pSwapchains = &m_swapchain;
    pi.pImageIndices = &imageIndex;
    vkQueuePresentKHR(m_presentQueue, &pi);

    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT; // advance to next frame slot
}

/// <summary>
/// Normalized [0,1] position along one monitor axis -> Vulkan NDC, snapped to a pixel centre
/// so shapes that have 'odd' sizing still look good
/// </summary>
static float normToNdc(float n, int sizePx)
{
    const float px = std::floor(n * static_cast<float>(sizePx)) + 0.5f;
    return 2.0f * px / static_cast<float>(sizePx) - 1.0f;
}

/// <summary>
/// Maps NDC to ONE monitor of the double-wide window.
/// xOffset 0 = left monitor, xOffset = monitorWidth = right monitor.
/// Everything drawn afterwards uses the same NDC on both monitors.
/// </summary>
void Renderer::setMonitorViewport(VkCommandBuffer cmd, int xOffset)
{
    VkViewport vp{};
    vp.x = static_cast<float>(xOffset);
    vp.y = 0.0f;
    vp.width = static_cast<float>(m_monitorWidth);
    vp.height = static_cast<float>(m_monitorHeight);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);

    VkRect2D sc{};
    sc.offset = { xOffset, 0 };
    sc.extent = { static_cast<uint32_t>(m_monitorWidth), static_cast<uint32_t>(m_monitorHeight) };
    vkCmdSetScissor(cmd, 0, 1, &sc);
}

/// <summary>
/// Recieves a frame scene object and translates it to Vulkan commands.
/// </summary>
/// <param name="cmd"> Command buffer </param>
/// <param name="imageIndex"></param>
/// <param name="scene"> Frame Scene object </param>
void Renderer::recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex, const FrameScene& scene)
{
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &bi); // begin the command buffer, start recording

    // render pass (clears frame buffer to background color)
    VkClearValue clearColor{};
    clearColor.color = {
     0.005182f,
     0.005605f,
     0.006995f,
     1.0f
    }; // im using a nice dark midnight blue for this calibration..
    VkRenderPassBeginInfo rpi{};
    rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpi.renderPass = m_renderPass;
    rpi.framebuffer = m_framebuffers[imageIndex];
    rpi.renderArea.offset = { 0, 0 };
    rpi.renderArea.extent = m_swapExtent;
    rpi.clearValueCount = 1;
    rpi.pClearValues = &clearColor;

    vkCmdBeginRenderPass(cmd, &rpi, VK_SUBPASS_CONTENTS_INLINE);

    // full screen NDC rect
    const float FX0 = -1.0f, FY0 = -1.0f, FX1 = 1.0f, FY1 = 1.0f;

    // one full screen image per monitor (the left/right textures are already mirrored at decode time)
    auto drawFullScreenImages = [&](TextureSlot left, TextureSlot right) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
        setMonitorViewport(cmd, 0);
        drawQuad(cmd, left, FX0, FY0, FX1, FY1);
        setMonitorViewport(cmd, m_monitorWidth);
        drawQuad(cmd, right, FX0, FY0, FX1, FY1);
        };

    // choose what to display based on the mode passed
    switch (scene.mode) {
    case CalibrationStage::StartInstructions:
        drawFullScreenImages(TEX_START, TEX_START);
        break;

    case CalibrationStage::HeadPositioning:
        renderHeadGuide(cmd, scene.head); // oval + eye circles, green / red
        break;

    case CalibrationStage::Calibration:
        renderCalibrationDots(cmd, scene.dots); // the current target, as a white dot
        break;

    case CalibrationStage::Calculating:
        // clear already handled by render pass
        drawFullScreenImages(TEX_CALCULATING, TEX_CALCULATING);
        break;

    case CalibrationStage::Failure:
        drawFullScreenImages(TEX_FAIL, TEX_FAIL);
        break;

    case CalibrationStage::Complete:
        // painter's order, back to front: success image -> fixation crosses -> gaze dots
        drawFullScreenImages(TEX_SUCCESS, TEX_SUCCESS);
        renderFixationPoints(cmd, scene.crosses); // every target (crosses)
        if (scene.drawGazeDots) renderGazeDots(cmd); // gaze samples on top of the crosses
        break;

    case CalibrationStage::Done:
        drawFullScreenImages(TEX_SAVING, TEX_SAVING);
        break;
    }

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

/// <summary>
/// Draws a single quad (textured rectangle) from 2 corners, (x0,y0) -> (x1,y1)
/// </summary>
/// <param name="cmd"></param>
/// <param name="slot">texture slot</param>
/// <param name="x0">top left corner, x</param>
/// <param name="y0">top left corner, y</param>
/// <param name="x1">bottom right corner, x</param>
/// <param name="y1">bottom right corner,y</param>
void Renderer::drawQuad(VkCommandBuffer cmd, TextureSlot slot, float x0, float y0, float x1, float y1)
{
    // texture selection -> tells fragment shader whcih image to sample from
    vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_pipelineLayout,
        0,
        1,
        &m_descSets[slot],
        0,
        nullptr
    );

    // send draw data to shaders (push constants is a quick ans small way to push ddata to shaders without ussing a buffer)
    QuadPushConstants pc{ x0, y0, x1, y1, (uint32_t)slot };
    vkCmdPushConstants(
        cmd,
        m_pipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(pc),
        &pc
    );

    // 1 quad = 2 triangles = 6 vertices
    // issue the draw call, per vertex 
    vkCmdDraw(cmd, 6, 1, 0, 0);
}


/// <summary>
/// Draws a fixation cross at each given position, on both monitors.
/// Positions are normalized display-area coordinates (Tobii space): (0,0) = top-left of the
/// physical monitor
/// 
/// ******* They are NOT mirrored --- TEST IF THIS MAKES SENSE. I think they shouldd be mirrored
/// </summary>
void Renderer::renderFixationPoints(VkCommandBuffer cmd, const std::vector<NormalizedPoint2D>& points)
{
    if (points.empty()) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_fixationPointPipeline);

    VkBuffer vbs[] = { m_fixationPointVB };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);

    const float halfSizeX = CROSSHAIR_SIZE_PX / static_cast<float>(m_monitorWidth);
    const float halfSizeY = CROSSHAIR_SIZE_PX / static_cast<float>(m_monitorHeight);

    for (int xOff : { 0, m_monitorWidth }) {
        setMonitorViewport(cmd, xOff);

        //for (const NormalizedPoint2D& p : points) {
        //    FixationPointPushConstants pc{};
        //    pc.centerX = normToNdc(p.x, m_monitorWidth);
        //    pc.centerY = normToNdc(p.y, m_monitorHeight);
        //    pc.halfSizeX = halfSizeX;
        //    pc.halfSizeY = halfSizeY;

        //    vkCmdPushConstants(cmd, m_fixationPointLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
        //    vkCmdDraw(cmd, 6, 1, 0, 0); // horizontal bar
        //    vkCmdDraw(cmd, 6, 1, 6, 0); // vertical bar
        //}

        // mirrored

        for (const NormalizedPoint2D& p : points) {
            FixationPointPushConstants pc{};
            pc.centerX = normToNdc(1.0f - p.x, m_monitorWidth);
            pc.centerY = normToNdc(p.y, m_monitorHeight);
            pc.halfSizeX = halfSizeX;
            pc.halfSizeY = halfSizeY;

            vkCmdPushConstants(cmd, m_fixationPointLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);

            vkCmdDraw(cmd, 6, 1, 0, 0);
            vkCmdDraw(cmd, 6, 1, 6, 0);
        }
    }
}

/// <summary>
/// Draws every dot uploaded via setGazeDots() on both monitors: one instanced draw per monitor.
/// </summary>
void Renderer::renderGazeDots(VkCommandBuffer cmd)
{
    if (m_gazeDotCount == 0) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_gazeDotPipeline);

    VkBuffer vbs[] = { m_gazeDotVB };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);

    GazeDotPushConstants pc{};
    pc.halfSizeX = 2.0f * GAZE_DOT_RADIUS_PX / static_cast<float>(m_monitorWidth);  // NDC spans 2 units
    pc.halfSizeY = 2.0f * GAZE_DOT_RADIUS_PX / static_cast<float>(m_monitorHeight);
    vkCmdPushConstants(cmd, m_gazeDotLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);

    for (int xOff : { 0, m_monitorWidth }) {
        setMonitorViewport(cmd, xOff);
        vkCmdDraw(cmd, 6, m_gazeDotCount, 0, 0); // 6 verts per dot, one instance per dot
    }
}

/// <summary>
/// Normalized [0,1] position -> pixel centre on one monitor 
/// so a dot sits exactly on the pixel Tobii calls (x, y).
/// </summary>
static float normToPixelCentre(float n, int sizePx)
{
    return std::floor(n * static_cast<float>(sizePx)) + 0.5f;
}

/// <summary>
/// Draws one anti-aliased shape (filled / outlined ellipse, or filled rect) on the current monitor viewport.
/// The caller binds m_shapePipeline and sets the monitor viewport first.
/// </summary>
void Renderer::drawShape(VkCommandBuffer cmd, ShapeKind kind,
    float cxPx, float cyPx, float halfWPx, float halfHPx,
    const float* rgb, float alpha, float outlinePx)
{
    const float W = static_cast<float>(m_monitorWidth);
    const float H = static_cast<float>(m_monitorHeight);

    ShapePushConstants pc{};
    pc.centerX = 2.0f * cxPx / W - 1.0f; // NDC: +Y is down in Vulkan
    pc.centerY = 2.0f * cyPx / H - 1.0f;
    pc.halfSizeX = 2.0f * halfWPx / W;
    pc.halfSizeY = 2.0f * halfHPx / H;
    pc.r = rgb[0];
    pc.g = rgb[1];
    pc.b = rgb[2];
    pc.a = alpha;
    // same pixel thickness on both axes, even for an oval
    pc.innerX = (outlinePx > 0.0f) ? (std::max)(0.0f, 1.0f - outlinePx / halfWPx) : 0.0f;
    pc.innerY = (outlinePx > 0.0f) ? (std::max)(0.0f, 1.0f - outlinePx / halfHPx) : 0.0f;
    pc.shape = (kind == ShapeKind::Rect) ? 1.0f : 0.0f;

    vkCmdPushConstants(cmd, m_shapeLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(pc), &pc);
    vkCmdDraw(cmd, 6, 1, 0, 0);
}

/// <summary>
/// Draws a white dot at each given position, on both monitors (the calibration target).
/// Positions are normalized display area coordinates (Tobii space) and are NOT mirrored
/// </summary>
void Renderer::renderCalibrationDots(VkCommandBuffer cmd, const std::vector<NormalizedPoint2D>& points)
{
    if (points.empty()) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shapePipeline);

    static const float WHITE[3] = { 1.0f, 1.0f, 1.0f };
    const float radius = 0.5f * CALIBRATION_DOT_DIAMETER_PX;

    for (int xOff : { 0, m_monitorWidth }) {
        setMonitorViewport(cmd, xOff);

        for (const NormalizedPoint2D& p : points) {
            drawShape(  cmd,  ShapeKind::Ellipse,   normToPixelCentre(1.0f - p.x, m_monitorWidth), normToPixelCentre(p.y, m_monitorHeight),   radius,  radius, WHITE,  1.0f  );
        }
    }
}

/// <summary>
/// Head positioning guide, drawn identically on both monitors.
/// layout: a head shaped oval in the middle; two target rings at eye level,
/// the participant's eyes as filled circles, and a depth gauge underneath.
/// 
/// Everything that signals state (oval outline, eye circles, gauge marker) is green when the head is in the right
/// place and red otherwise; with no valid eyes only the red oval and rings are shown.
/// </summary>
void Renderer::renderHeadGuide(VkCommandBuffer cmd, const HeadGuide& g)
{
    const float W = static_cast<float>(m_monitorWidth);
    const float H = static_cast<float>(m_monitorHeight);
    const float* state = g.inPosition ? GUIDE_GOOD_RGB : GUIDE_BAD_RGB;

    const float cx = 0.5f * W;
    const float cy = 0.5f * H;
    const float ovalRx = HEAD_OVAL_RX * H;
    const float ovalRy = HEAD_OVAL_RY * H;
    const float eyeLineY = cy + HEAD_EYE_LINE_OFFSET * H;
    const float ringR = HEAD_EYE_RING_RADIUS * H;
    const float rangeX = HEAD_RANGE_X * H;
    const float rangeY = HEAD_RANGE_Y * H;
    const float ringDx = HEAD_GUIDE_IDEAL_EYE_HALF_SEPARATION * rangeX;
    const float gaugeY = cy + HEAD_GAUGE_Y * H;
    const float gaugeHalfW = HEAD_GAUGE_HALF_WIDTH * H;

    // x laid out as the participant should see it -> x on the physical monitor
    auto physX = [&](float viewX) { return  (W - viewX); };

    // eye position (participant's view, 0..1 across the track box) -> pixels in the participant's view
    auto eyeX = [&](const HeadGuideEye& e) { return cx + (std::clamp(e.x, 0.0f, 1.0f) - 0.5f) * rangeX; };
    auto eyeY = [&](const HeadGuideEye& e) { return eyeLineY + (std::clamp(e.y, 0.0f, 1.0f) - 0.5f) * rangeY; };

    // too close (z < 0.5) -> bigger, too far (z > 0.5) -> smaller; matches the ring at the ideal distance
    auto eyeRadius = [&](const HeadGuideEye& e) {
        return ringR * HEAD_EYE_DOT_SCALE * std::clamp(1.0f + (0.5f - e.z) * HEAD_DEPTH_GAIN, 0.4f, 1.8f);
        };

    // depth gauge marker: mean depth of whichever eyes are valid
    float zSum = 0.0f;
    int   zCount = 0;
    for (const HeadGuideEye* e : { &g.left, &g.right })
        if (e->valid) { zSum += e->z; ++zCount; }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shapePipeline);

    for (int xOff : { 0, m_monitorWidth }) {
        setMonitorViewport(cmd, xOff);

        // head: faint tint + outline
        drawShape(cmd, ShapeKind::Ellipse, physX(cx), cy, ovalRx, ovalRy, state, 0.10f);
        drawShape(cmd, ShapeKind::Ellipse, physX(cx), cy, ovalRx, ovalRy, state, 1.0f, HEAD_OVAL_OUTLINE_PX);

        // where the eyes should be
        drawShape(cmd, ShapeKind::Ellipse, physX(cx - ringDx), eyeLineY, ringR, ringR,
            GUIDE_NEUTRAL_RGB, 1.0f, HEAD_EYE_RING_OUTLINE_PX);
        drawShape(cmd, ShapeKind::Ellipse, physX(cx + ringDx), eyeLineY, ringR, ringR,
            GUIDE_NEUTRAL_RGB, 1.0f, HEAD_EYE_RING_OUTLINE_PX);

        // depth gauge: track, target zone, marker
        const float zoneHalfW = g.depthTolerance * 2.0f * gaugeHalfW;
        drawShape(cmd, ShapeKind::Rect, physX(cx), gaugeY, gaugeHalfW, 0.006f * H, GUIDE_NEUTRAL_RGB, 0.6f);
        drawShape(cmd, ShapeKind::Rect, physX(cx), gaugeY, zoneHalfW, 0.012f * H, GUIDE_GOOD_RGB, 0.8f);
        if (zCount > 0) {
            const float z = std::clamp(zSum / static_cast<float>(zCount), 0.0f, 1.0f);
            const float markerX = cx - gaugeHalfW + z * 2.0f * gaugeHalfW; // left = too close, right = too far
            drawShape(cmd, ShapeKind::Rect, physX(markerX), gaugeY, 0.004f * H, 0.03f * H, state, 1.0f);
        }

        // the participant's eyes on top
        for (const HeadGuideEye* e : { &g.left, &g.right }) {
            if (!e->valid) continue;
            const float r = eyeRadius(*e);
            drawShape(cmd, ShapeKind::Ellipse, physX(eyeX(*e)), eyeY(*e), r, r, state, 1.0f);
        }
    }
}

/// <summary>
/// Uploads the gaze dots for the Complete screen. Positions are normalized display area
/// coordinates, converted here to NDC (Vulkan: +Y down, so ndc = 2n - 1 on both axes).
/// Waits for the GPU first (in flight frames)
/// </summary>
void Renderer::setGazeDots(const std::vector<GazeDot>& dots)
{
    if (dots.size() > MAX_GAZE_DOTS)
        std::cerr << "[Renderer] " << dots.size() << " gaze dots requested, only drawing the first "
        << MAX_GAZE_DOTS << "\n";

    const uint32_t n = static_cast<uint32_t>((std::min)(dots.size(), static_cast<size_t>(MAX_GAZE_DOTS)));

    vkDeviceWaitIdle(m_device);

    auto* out = static_cast<GazeDotInstance*>(m_gazeDotMapped);
    for (uint32_t i = 0; i < n; ++i) {
        const GazeDot& d = dots[i];
        out[i] = { 2.0f * d.x - 1.0f, 2.0f * d.y - 1.0f, d.r, d.g, d.b, d.a };
    }
    m_gazeDotCount = n;
}

void Renderer::clearGazeDots()
{
    m_gazeDotCount = 0; // only read while recording on the same thread, so no GPU sync needed
}


DecodedImage Renderer::decodeImageForUpload(TextureSlot slot, const std::string& path) {

    // IMREAD_COLOR returns 8-bit BGR
    cv::Mat src = cv::imread(path, cv::IMREAD_COLOR);
    if (src.empty()) throw std::runtime_error("[Renderer] Failed to load image: " + path);

    // images need to be mirrored (stereoscopic mirror setup), and for some reason they're loaded in upside down,
    // so flip on both axes (x mirrors for the setup, y corrects the upside down load)
    cv::flip(src, src, -1);

    // openCV uses BGR, vulkan expects RGB(A)
    cv::Mat rgba;
    cv::cvtColor(src, rgba, cv::COLOR_BGR2RGBA);
    if (!rgba.isContinuous()) rgba = rgba.clone();

    DecodedImage out;
    out.format = VK_FORMAT_R8G8B8A8_SRGB; // the PPM values are already sRGB encoded; the GPU linearizes them when sampling
    out.width = rgba.cols;
    out.height = rgba.rows;
    const size_t bytes = rgba.total() * rgba.elemSize();
    out.pixels.assign(rgba.data, rgba.data + bytes);

    m_decodedImages[slot] = out;
    return out;
}


void Renderer::uploadDecodedTexture(TextureSlot slot) {
    // wait for the gpu to finish any in flight frame before destroying/replacing this slot's texture
    // destroyTexture() has no synchronization of its own, and a previous frames command buffer
    // may still be referencing the old texture (MAX_FRAMES_IN_FLIGHT > 1).
    vkDeviceWaitIdle(m_device);

    Texture& tex = m_textures[slot];
    DecodedImage& decodedImage = m_decodedImages[slot];
    destroyTexture(tex);
    tex.width = decodedImage.width;
    tex.height = decodedImage.height;
    uploadTextureData(tex, decodedImage.pixels.data(), decodedImage.format, decodedImage.width, decodedImage.height);
    updateDescriptorSet(slot, tex);
}

void Renderer::decodeAndUploadTexture(TextureSlot slot, const std::string& path) {
    decodeImageForUpload(slot, path);
    uploadDecodedTexture(slot);  // now waits for GPU idle internally
}



/// <summary>
/// Staging buffer upload. 
/// Staging buffers are used to efficiently transfer data from CPU -> GPU. A bridge.
/// </summary>
/// <param name="tex"></param>
/// <param name="pixels"></param>
/// <param name="fmt"></param>
/// <param name="w"></param>
/// <param name="h"></param>
void Renderer::uploadTextureData(Texture& tex, const void* pixels,
    VkFormat fmt, int w, int h)
{
    //create staging buffer in host visible memory
    const size_t bpp = (fmt == VK_FORMAT_R16G16B16A16_SFLOAT) ? 8 : 4;
    const VkDeviceSize size = (VkDeviceSize)(w * h) * bpp;

    VkBuffer stageBuf;
    VkDeviceMemory stageMem;
    createBuffer(
        size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stageBuf,
        stageMem);

    void* mapped;
    // map gpu memory into cpu address space to write into it directly
    vkMapMemory(m_device, stageMem, 0, size, 0, &mapped);
    //memcpy the pixel data 
    std::memcpy(mapped, pixels, size);
    //then unmap it
    vkUnmapMemory(m_device, stageMem);

    // gpu image
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent = { (uint32_t)w, (uint32_t)h, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL; // optimize for gpu, cpu can no longer read it
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT; // image wil recieve data from sstaging buffer and be sampled in shaders
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    vkCreateImage(m_device, &ici, nullptr, &tex.image);

    // get the memory requirements for the image
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(m_device, tex.image, &mr);

    // then allocate the memory (fast GPU memory, not accessible by CPU)
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(m_device, &mai, nullptr, &tex.memory);

    // bind the allocated memory to image
    vkBindImageMemory(m_device, tex.image, tex.memory, 0);

    // transition image to Transfer_DST_OPTIMAL state to tell vulkan we are about to write into it
    transitionImageLayout(tex.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // copy the staging buffer into gpu image
    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { (uint32_t)w, (uint32_t)h, 1 };

    // copies raw pixel data from CPU staging buffer into GPU image memory
    vkCmdCopyBufferToImage(cmd, stageBuf, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    endSingleTimeCommands(cmd);

    // image state, ready for shaders
    transitionImageLayout(tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // image view -> how shaders will interpret this image
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = tex.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    // define how vulkan will interpret the image data (color channels, mip levels etc)
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.layerCount = 1;

    vkCreateImageView(m_device, &vci, nullptr, &tex.view);

    // sampler (how the texture is read in shaders)
    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

    // linear filtering smooth the interpolation between texels
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;

    // clamp prevents from wrapping at edges
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    vkCreateSampler(m_device, &sci, nullptr, &tex.sampler);

    // cleanup temp upload resources (reverse creation order)
    vkDestroyBuffer(m_device, stageBuf, nullptr);
    vkFreeMemory(m_device, stageMem, nullptr);
}

/// <summary>
/// Helper to destroy textures
/// </summary>
/// <param name="tex"></param>
void Renderer::destroyTexture(Texture& tex) {
    if (!m_device) return;
    if (tex.sampler) { vkDestroySampler(m_device, tex.sampler, nullptr); tex.sampler = VK_NULL_HANDLE; }
    if (tex.view) { vkDestroyImageView(m_device, tex.view, nullptr);   tex.view = VK_NULL_HANDLE; }
    if (tex.image) { vkDestroyImage(m_device, tex.image, nullptr);      tex.image = VK_NULL_HANDLE; }
    if (tex.memory) { vkFreeMemory(m_device, tex.memory, nullptr);       tex.memory = VK_NULL_HANDLE; }
}

/// <summary>
/// Helper to update the descriptor set
/// </summary>
/// <param name="slot"></param>
/// <param name="tex"></param>
void Renderer::updateDescriptorSet(int slot, const Texture& tex) {
    VkDescriptorImageInfo ii{};
    ii.sampler = tex.sampler;
    ii.imageView = tex.view;
    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet wr{};
    wr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr.dstSet = m_descSets[slot];
    wr.dstBinding = 0;
    wr.descriptorCount = 1;
    wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr.pImageInfo = &ii;
    vkUpdateDescriptorSets(m_device, 1, &wr, 0, nullptr);
}

// Vulkan setup

/// <summary>
/// create the root vulkan object
/// </summary>
void Renderer::createInstance() {
    if (ENABLE_VALIDATION && !checkValidationLayerSupport())
        throw std::runtime_error("Validation layers requested but not available");

    VkApplicationInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "FlickerExperiment";
    ai.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    ai.apiVersion = VK_API_VERSION_1_2;

    auto exts = getRequiredExtensions();

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &ai;
    ci.enabledExtensionCount = (uint32_t)exts.size();
    ci.ppEnabledExtensionNames = exts.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCI{};
    if (ENABLE_VALIDATION) {
        ci.enabledLayerCount = (uint32_t)VALIDATION_LAYERS.size();
        ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();

        debugCI.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCI.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugCI.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
        debugCI.pfnUserCallback = debugCallback;
        ci.pNext = &debugCI;
    }

    if (vkCreateInstance(&ci, nullptr, &m_instance) != VK_SUCCESS)
        throw std::runtime_error("vkCreateInstance failed");
}
/// <summary>
/// Debug callbacks
/// </summary>
void Renderer::setupDebugMessenger() {
    VkDebugUtilsMessengerCreateInfoEXT ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    ci.pfnUserCallback = debugCallback;

    auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT");
    if (!fn) throw std::runtime_error("Cannot load vkCreateDebugUtilsMessengerEXT");
    fn(m_instance, &ci, nullptr, &m_debugMessenger);
}

/// <summary>
/// Creates the drawing surface from a GLFW window
/// </summary>
/// <param name="window"></param>
void Renderer::createSurface(GLFWwindow* window) {
    if (glfwCreateWindowSurface(m_instance, window, nullptr, &m_surface) != VK_SUCCESS)
        throw std::runtime_error("glfwCreateWindowSurface failed");
}

/// <summary>
/// Pick a Vulkan comaptible GPU to use
/// </summary>
void Renderer::pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
    if (!count) throw std::runtime_error("No Vulkan compatible GPU found");

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(m_instance, &count, devices.data());

    for (auto& dev : devices) {
        auto idx = findQueueFamilies(dev);
        auto scs = querySwapChainSupport(dev);
        if (idx.isComplete() && !scs.formats.empty() && !scs.presentModes.empty()) {
            m_physicalDevice = dev;
            break;
        }
    }
    if (!m_physicalDevice) throw std::runtime_error("No suitable GPU found");
}

/// <summary>
/// Create the software to handle to that GPU + queues
/// </summary>
void Renderer::createLogicalDevice() {
    auto idx = findQueueFamilies(m_physicalDevice);
    std::set<uint32_t> uniqueFamilies{ idx.graphics.value(), idx.present.value() };

    float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> qcis;
    for (auto family : uniqueFamilies) {
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;
        qcis.push_back(qci);
    }

    VkPhysicalDeviceFeatures features{};

    VkDeviceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.queueCreateInfoCount = (uint32_t)qcis.size();
    ci.pQueueCreateInfos = qcis.data();
    ci.enabledExtensionCount = (uint32_t)DEVICE_EXTENSIONS.size();
    ci.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();
    ci.pEnabledFeatures = &features;

    if (ENABLE_VALIDATION) {
        ci.enabledLayerCount = (uint32_t)VALIDATION_LAYERS.size();
        ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
    }

    if (vkCreateDevice(m_physicalDevice, &ci, nullptr, &m_device) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDevice failed");

    vkGetDeviceQueue(m_device, idx.graphics.value(), 0, &m_graphicsQueue);
    vkGetDeviceQueue(m_device, idx.present.value(), 0, &m_presentQueue);
}


/// <summary>
/// Chooses the swap format in order of priority
/// HDR10 first, fall back on sRGB
/// </summary>
/// <param name="available"></param>
/// <returns></returns>
VkSurfaceFormatKHR Renderer::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available)
{

    for (auto& f : available) { // SRGB only

        if (f.format == VK_FORMAT_R16G16B16A16_SFLOAT &&
            f.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT) return f;
    }

    return available[0];
}

/// <summary>
/// Enables v sync
/// </summary>
/// <param name="available"></param>
/// <returns></returns>
VkPresentModeKHR Renderer::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& available)
{
    //for (auto m : available)
    //    if (m == VK_PRESENT_MODE_FIFO_KHR) return m;
    return VK_PRESENT_MODE_FIFO_KHR;
}

/// <summary>
/// Choose the swap chain resolution
/// </summary>
/// <param name="window"></param>
/// <param name="caps"></param>
/// <returns></returns>
VkExtent2D Renderer::chooseSwapExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& caps)
{
    if (caps.currentExtent.width != UINT32_MAX) return caps.currentExtent;

    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    return {
        std::clamp((uint32_t)w, caps.minImageExtent.width,  caps.maxImageExtent.width),
        std::clamp((uint32_t)h, caps.minImageExtent.height, caps.maxImageExtent.height)
    };
}



/// <summary>
/// Creates the swap chain (with helpers)
/// </summary>
/// <param name="window"></param>
void Renderer::createSwapChain(GLFWwindow* window) {
    auto scs = querySwapChainSupport(m_physicalDevice);
    auto sfmt = chooseSwapSurfaceFormat(scs.formats);
    auto smode = chooseSwapPresentMode(scs.presentModes);
    auto extent = chooseSwapExtent(window, scs.capabilities);

    m_swapFormat = sfmt.format;
    m_swapColorSpace = sfmt.colorSpace;
    m_swapExtent = extent;

    uint32_t imgCount = scs.capabilities.minImageCount + 1;
    if (scs.capabilities.maxImageCount > 0)
        imgCount = (std::min)(imgCount, scs.capabilities.maxImageCount);

    VkSwapchainCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = m_surface;
    ci.minImageCount = imgCount;
    ci.imageFormat = sfmt.format;
    ci.imageColorSpace = sfmt.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    auto idx = findQueueFamilies(m_physicalDevice);
    uint32_t families[] = { idx.graphics.value(), idx.present.value() };
    if (idx.graphics != idx.present) {
        ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices = families;
    }
    else {
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    ci.preTransform = scs.capabilities.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = smode;
    ci.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(m_device, &ci, nullptr, &m_swapchain) != VK_SUCCESS)
        throw std::runtime_error("vkCreateSwapchainKHR failed");

    // retrieve swap chain images
    uint32_t n;
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &n, nullptr);
    m_swapImages.resize(n);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &n, m_swapImages.data());
}

void Renderer::createImageViews() {
    m_swapImageViews.resize(m_swapImages.size());
    for (size_t i = 0; i < m_swapImages.size(); ++i) {
        VkImageViewCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image = m_swapImages[i];
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = m_swapFormat;
        ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ci.subresourceRange.levelCount = 1;
        ci.subresourceRange.layerCount = 1;
        vkCreateImageView(m_device, &ci, nullptr, &m_swapImageViews[i]);
    }
}
/// <summary>
/// Render target layout transitions
/// </summary>
void Renderer::createRenderPass() {
    VkAttachmentDescription att{};
    att.format = m_swapFormat;
    att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference ref{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &ref;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpi{};
    rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpi.attachmentCount = 1;
    rpi.pAttachments = &att;
    rpi.subpassCount = 1;
    rpi.pSubpasses = &sub;
    rpi.dependencyCount = 1;
    rpi.pDependencies = &dep;

    if (vkCreateRenderPass(m_device, &rpi, nullptr, &m_renderPass) != VK_SUCCESS)
        throw std::runtime_error("vkCreateRenderPass failed");
}

void Renderer::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 1;
    li.pBindings = &b;

    if (vkCreateDescriptorSetLayout(m_device, &li, nullptr, &m_descSetLayout) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDescriptorSetLayout failed");
}

void Renderer::allocateDescriptorPool() {
    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ps.descriptorCount = MAX_TEXTURES;

    VkDescriptorPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.poolSizeCount = 1;
    pi.pPoolSizes = &ps;
    pi.maxSets = MAX_TEXTURES;

    if (vkCreateDescriptorPool(m_device, &pi, nullptr, &m_descPool) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDescriptorPool failed");

    std::vector<VkDescriptorSetLayout> layouts(MAX_TEXTURES, m_descSetLayout);
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = m_descPool;
    ai.descriptorSetCount = MAX_TEXTURES;
    ai.pSetLayouts = layouts.data();

    if (vkAllocateDescriptorSets(m_device, &ai, m_descSets.data()) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateDescriptorSets failed");
}

VkShaderModule Renderer::createShaderModule(const std::vector<uint32_t>& spirv) {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = spirv.size() * sizeof(uint32_t);
    ci.pCode = spirv.data();

    VkShaderModule mod;
    if (vkCreateShaderModule(m_device, &ci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("vkCreateShaderModule failed");
    return mod;
}

void Renderer::createGraphicsPipeline() {
    auto vertSpv = compileGLSL(QUAD_VERT_GLSL, shaderc_vertex_shader, "quad.vert");
    auto fragSpv = compileGLSL(QUAD_FRAG_GLSL, shaderc_fragment_shader, "quad.frag");

    VkShaderModule vertMod = createShaderModule(vertSpv);
    VkShaderModule fragMod = createShaderModule(fragSpv);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vis{};
    vis.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo ias{};
    ias.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ias.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vps{};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rast{};
    rast.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode = VK_CULL_MODE_NONE;
    rast.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAtt{};
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAtt.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAtt;

    std::vector<VkDynamicState> dynStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = (uint32_t)dynStates.size();
    dyn.pDynamicStates = dynStates.data();

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(QuadPushConstants);

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &m_descSetLayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcRange;

    vkCreatePipelineLayout(m_device, &pli, nullptr, &m_pipelineLayout);

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vis;
    pci.pInputAssemblyState = &ias;
    pci.pViewportState = &vps;
    pci.pRasterizationState = &rast;
    pci.pMultisampleState = &ms;
    pci.pColorBlendState = &blend;
    pci.pDynamicState = &dyn;
    pci.layout = m_pipelineLayout;
    pci.renderPass = m_renderPass;

    if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pci,
        nullptr, &m_pipeline) != VK_SUCCESS)
        throw std::runtime_error("vkCreateGraphicsPipelines failed");

    vkDestroyShaderModule(m_device, vertMod, nullptr);
    vkDestroyShaderModule(m_device, fragMod, nullptr);
}
/// <summary>
/// Shared setup for the overlay pipelines (cross, gaze dots, shapes): triangle list, no descriptor sets,
/// dynamic viewport + scissor, one push constant block (vertex stage only unless told otherwise),
/// optional alpha blending.
/// </summary>
void Renderer::createOverlayPipeline(
    const std::string& vertGlsl, const std::string& fragGlsl, const char* name,
    const VkPipelineVertexInputStateCreateInfo& vis,
    uint32_t pushConstantSize, bool alphaBlend,
    VkPipelineLayout& layoutOut, VkPipeline& pipelineOut,
    VkShaderStageFlags pushConstantStages)
{
    auto vertSpv = compileGLSL(vertGlsl, shaderc_vertex_shader, (std::string(name) + ".vert").c_str());
    auto fragSpv = compileGLSL(fragGlsl, shaderc_fragment_shader, (std::string(name) + ".frag").c_str());

    VkShaderModule vertMod = createShaderModule(vertSpv);
    VkShaderModule fragMod = createShaderModule(fragSpv);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName = "main";

    VkPipelineInputAssemblyStateCreateInfo ias{};
    ias.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ias.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vps{};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rast{};
    rast.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode = VK_CULL_MODE_NONE;
    rast.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAtt{};
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    if (alphaBlend) {
        blendAtt.blendEnable = VK_TRUE;
        blendAtt.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blendAtt.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAtt.colorBlendOp = VK_BLEND_OP_ADD;
        blendAtt.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAtt.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        blendAtt.alphaBlendOp = VK_BLEND_OP_ADD;
    }
    else {
        blendAtt.blendEnable = VK_FALSE;
    }

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAtt;

    std::vector<VkDynamicState> dynStates{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = (uint32_t)dynStates.size();
    dyn.pDynamicStates = dynStates.data();

    VkPushConstantRange pcr{};
    pcr.stageFlags = pushConstantStages;
    pcr.offset = 0;
    pcr.size = pushConstantSize;

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(m_device, &pli, nullptr, &layoutOut) != VK_SUCCESS)
        throw std::runtime_error(std::string("vkCreatePipelineLayout failed: ") + name);

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vis;
    pci.pInputAssemblyState = &ias;
    pci.pViewportState = &vps;
    pci.pRasterizationState = &rast;
    pci.pMultisampleState = &ms;
    pci.pColorBlendState = &blend;
    pci.pDynamicState = &dyn;
    pci.layout = layoutOut;
    pci.renderPass = m_renderPass;

    if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipelineOut) != VK_SUCCESS)
        throw std::runtime_error(std::string("vkCreateGraphicsPipelines failed: ") + name);

    vkDestroyShaderModule(m_device, vertMod, nullptr);
    vkDestroyShaderModule(m_device, fragMod, nullptr);
}

/// <summary>
/// Fixation cross: per-vertex 2D positions from m_fixationPointVB, placed by push constants.
/// </summary>
void Renderer::createFixationPointPipeline() {
    VkVertexInputBindingDescription bind{};
    bind.binding = 0;
    bind.stride = 2 * sizeof(float);
    bind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attr{};
    attr.location = 0;
    attr.binding = 0;
    attr.format = VK_FORMAT_R32G32_SFLOAT;
    attr.offset = 0;

    VkPipelineVertexInputStateCreateInfo vis{};
    vis.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vis.vertexBindingDescriptionCount = 1;
    vis.pVertexBindingDescriptions = &bind;
    vis.vertexAttributeDescriptionCount = 1;
    vis.pVertexAttributeDescriptions = &attr;

    createOverlayPipeline(CROSSHAIR_VERT_GLSL, CROSSHAIR_FRAG_GLSL, "crosshair", vis,
        sizeof(FixationPointPushConstants), /*alphaBlend*/ false,
        m_fixationPointLayout, m_fixationPointPipeline);
}

/// <summary>
/// Gaze dots: one INSTANCE per dot (centre + colour from m_gazeDotVB), quad built in the vertex shader.
/// Alpha blended, so samples the calibration discarded can be drawn faded.
/// </summary>
void Renderer::createGazeDotPipeline() {
    VkVertexInputBindingDescription bind{};
    bind.binding = 0;
    bind.stride = sizeof(GazeDotInstance);
    bind.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    VkVertexInputAttributeDescription attrs[2]{};
    attrs[0].location = 0; // centre (NDC)
    attrs[0].binding = 0;
    attrs[0].format = VK_FORMAT_R32G32_SFLOAT;
    attrs[0].offset = offsetof(GazeDotInstance, ndcX);
    attrs[1].location = 1; // colour
    attrs[1].binding = 0;
    attrs[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attrs[1].offset = offsetof(GazeDotInstance, r);

    VkPipelineVertexInputStateCreateInfo vis{};
    vis.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vis.vertexBindingDescriptionCount = 1;
    vis.pVertexBindingDescriptions = &bind;
    vis.vertexAttributeDescriptionCount = 2;
    vis.pVertexAttributeDescriptions = attrs;

    createOverlayPipeline(GAZE_DOT_VERT_GLSL, GAZE_DOT_FRAG_GLSL, "gazedot", vis,
        sizeof(GazeDotPushConstants), /*alphaBlend*/ true,
        m_gazeDotLayout, m_gazeDotPipeline);
}

/// <summary>
/// Shapes: no vertex buffers at all, the quad is generated in the vertex shader from push constants.
/// Alpha blended (anti-aliased edges, tinted fills). Push constants are read by both stages.
/// </summary>
void Renderer::createShapePipeline() {
    VkPipelineVertexInputStateCreateInfo vis{};
    vis.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    createOverlayPipeline(SHAPE_VERT_GLSL, SHAPE_FRAG_GLSL, "shape", vis,
        sizeof(ShapePushConstants), /*alphaBlend*/ true,
        m_shapeLayout, m_shapePipeline,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
}

void Renderer::createFramebuffers() {
    m_framebuffers.resize(m_swapImageViews.size());
    for (size_t i = 0; i < m_swapImageViews.size(); ++i) {
        VkFramebufferCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fi.renderPass = m_renderPass;
        fi.attachmentCount = 1;
        fi.pAttachments = &m_swapImageViews[i];
        fi.width = m_swapExtent.width;
        fi.height = m_swapExtent.height;
        fi.layers = 1;
        vkCreateFramebuffer(m_device, &fi, nullptr, &m_framebuffers[i]);
    }
}

void Renderer::createFixationPointVertexBuffer() {
    const VkDeviceSize size = sizeof(CROSSHAIR_VERTS);

    VkBuffer       stageBuf;
    VkDeviceMemory stageMem;
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stageBuf, stageMem);

    void* mapped;
    vkMapMemory(m_device, stageMem, 0, size, 0, &mapped);
    std::memcpy(mapped, CROSSHAIR_VERTS, size);
    vkUnmapMemory(m_device, stageMem);

    createBuffer(size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        m_fixationPointVB, m_fixationPointVBMem);

    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkBufferCopy    cp{ 0, 0, size };
    vkCmdCopyBuffer(cmd, stageBuf, m_fixationPointVB, 1, &cp);
    endSingleTimeCommands(cmd);

    vkDestroyBuffer(m_device, stageBuf, nullptr);
    vkFreeMemory(m_device, stageMem, nullptr);
}

/// <summary>
/// Host-visible, persistently mapped buffer for the gaze dot instances.
/// Written rarely (once per calibration result) so no staging buffer is needed.
/// </summary>
void Renderer::createGazeDotBuffer() {
    const VkDeviceSize size = sizeof(GazeDotInstance) * MAX_GAZE_DOTS;
    createBuffer(size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_gazeDotVB, m_gazeDotVBMem);
    vkMapMemory(m_device, m_gazeDotVBMem, 0, size, 0, &m_gazeDotMapped);
}


void Renderer::createCommandPool() {
    auto idx = findQueueFamilies(m_physicalDevice);
    VkCommandPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.queueFamilyIndex = idx.graphics.value();
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(m_device, &ci, nullptr, &m_commandPool);
}

void Renderer::createCommandBuffers() {
    m_commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = m_commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
    vkAllocateCommandBuffers(m_device, &ai, m_commandBuffers.data());
}

VkCommandBuffer Renderer::beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = m_commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(m_device, &ai, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

void Renderer::endSingleTimeCommands(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    vkQueueSubmit(m_graphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_graphicsQueue);
    vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmd);
}

void Renderer::createSyncObjects() {
    VkSemaphoreCreateInfo si{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkFenceCreateInfo     fi{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vkCreateSemaphore(m_device, &si, nullptr, &m_imageAvailable[i]);
        vkCreateSemaphore(m_device, &si, nullptr, &m_renderFinished[i]);
        vkCreateFence(m_device, &fi, nullptr, &m_inFlightFences[i]);
    }
}


//************* Buffer and Memory helpers 
void Renderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem)
{
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(m_device, &bi, nullptr, &buf);

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(m_device, buf, &mr);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, props);
    vkAllocateMemory(m_device, &ai, nullptr, &mem);
    vkBindBufferMemory(m_device, buf, mem, 0);
}

uint32_t Renderer::findMemoryType(uint32_t filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((filter & (1 << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("findMemoryType: no suitable type found");
}

void Renderer::transitionImageLayout(VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout)
{
    VkCommandBuffer cmd = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkPipelineStageFlags srcStage, dstStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
        newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else {
        throw std::runtime_error("Unsupported layout transition");
    }

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    endSingleTimeCommands(cmd);
}

// ****************** queries

QueueFamilyIndices Renderer::findQueueFamilies(VkPhysicalDevice dev) {
    QueueFamilyIndices idx;
    uint32_t count;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, props.data());

    for (uint32_t i = 0; i < count; ++i) {
        if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            idx.graphics = i;

        VkBool32 present = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, m_surface, &present);
        if (present) idx.present = i;

        if (idx.isComplete()) break;
    }
    return idx;
}
/// <summary>
/// Query the GPU device for supported swapchain format-color space pairs 
/// </summary>
/// <param name="dev"></param>
/// <returns></returns>
SwapChainSupportDetails Renderer::querySwapChainSupport(VkPhysicalDevice dev) {
    SwapChainSupportDetails d;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, m_surface, &d.capabilities);
    uint32_t n;
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev, m_surface, &n, nullptr);
    d.formats.resize(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev, m_surface, &n, d.formats.data());

    vkGetPhysicalDeviceSurfacePresentModesKHR(dev, m_surface, &n, nullptr);
    d.presentModes.resize(n);
    vkGetPhysicalDeviceSurfacePresentModesKHR(dev, m_surface, &n, d.presentModes.data());

    return d;
}

// *********************** validation helpers

bool Renderer::checkValidationLayerSupport() {
    uint32_t count;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> available(count);
    vkEnumerateInstanceLayerProperties(&count, available.data());

    for (const char* name : VALIDATION_LAYERS) {
        bool found = false;
        for (auto& layer : available)
            if (std::strcmp(name, layer.layerName) == 0) { found = true; break; }
        if (!found) return false;
    }
    return true;
}

std::vector<const char*> Renderer::getRequiredExtensions() {
    uint32_t glfwCount;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwCount);
    std::vector<const char*> exts(glfwExts, glfwExts + glfwCount);
    exts.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
    if (ENABLE_VALIDATION)
        exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    return exts;
}

VKAPI_ATTR VkBool32 VKAPI_CALL Renderer::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*)
{
    std::cerr << "[VK] " << data->pMessage << "\n";
    return VK_FALSE;
}