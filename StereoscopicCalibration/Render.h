#pragma once

#include "rendertypes.h"

#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <array>
#include <optional>
#include <string>
#include <vector>

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;
    bool isComplete() const { return graphics.has_value() && present.has_value(); }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR        capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   presentModes;
};
struct DisplayColorInfo {
    // chromaticities (CIE 1931 xy), matches VkXYColorEXT layout
    float redPrimary[2];
    float greenPrimary[2];
    float bluePrimary[2];
    float whitePoint[2];
    float minLuminance; // nits
    float maxLuminance; // nits (peak, small area)
    float maxFullFrameLuminance; // nits (sustained full-screen)
    bool  isHDR; // true if color space reports PQ/BT.2100
    bool  valid; // false if query failed
};
class Renderer {
public:
    Renderer() = default;
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    bool init(GLFWwindow* window, int monitorWidth, int monitorHeight, int displayMode = 0);

    void decodeAndUploadTexture(TextureSlot slot, const std::string& path);

    // render & present 1 frame. blocks on vsync (FIFO)
    void drawFrame(const FrameScene& scene);

    // call before app destroys the window or releases shared resources
    void waitIdle();

    DecodedImage decodeImageForUpload(TextureSlot slot, const std::string& path);

    //// GPU only. can only be called from main renderer thread
    void uploadDecodedTexture(TextureSlot slot);

    // Upload the gaze dots drawn on the Complete screen (when FrameScene::drawGazeDots is true).
    // Dot positions are normalized display-area coordinates (Tobii space, NOT mirrored).
    // Capped at MAX_GAZE_DOTS. Main renderer thread only.
    void setGazeDots(const std::vector<GazeDot>& dots);
    void clearGazeDots();

    static constexpr uint32_t MAX_GAZE_DOTS = 16384;

private:
    //  per slot texture resourcess
    struct Texture {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        uint32_t width = 0;
        uint32_t height = 0;
    };

    // push constants for the quad pipeline
    struct QuadPushConstants {
        float x0, y0, x1, y1;
        uint32_t texSlot;
    };

    // push constants for the fixation cross pipeline
    struct FixationPointPushConstants {
        float centerX;
        float centerY;
        float halfSizeX;
        float halfSizeY;
    };

    // push constants for the gaze dot pipeline
    struct GazeDotPushConstants {
        float halfSizeX;
        float halfSizeY;
    };

    // per-instance vertex data for the gaze dot pipeline (24 bytes)
    struct GazeDotInstance {
        float ndcX, ndcY;
        float r, g, b, a;
    };

    // push constants for the shape pipeline (white dots, head guide). All floats on purpose: no std430 alignment surprises.
    struct ShapePushConstants {
        float centerX, centerY;     // NDC
        float halfSizeX, halfSizeY; // NDC
        float r, g, b, a;
        float innerX, innerY;       // outline inner radius / outer radius per axis (0 = filled)
        float shape;                // 0 = ellipse, 1 = rect
    };

    enum class ShapeKind { Ellipse, Rect };

    // vulkan init chain
    void createInstance();
    void setupDebugMessenger();
    void createSurface(GLFWwindow* window);
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain(GLFWwindow* window);
    void createImageViews();
    void createRenderPass();
    void createDescriptorSetLayout();
    void allocateDescriptorPool();
    void createGraphicsPipeline();       // textured quads
    void createFixationPointPipeline();  // cross
    void createGazeDotPipeline();        // instanced dots
    void createShapePipeline();          // white calibration dots + head positioning guide
    void createFramebuffers();
    void createFixationPointVertexBuffer();
    void createGazeDotBuffer();
    void createCommandPool();
    void createCommandBuffers();
    void createSyncObjects();

    // shared boilerplate for the overlay pipelines (no descriptor sets, dynamic viewport/scissor)
    void createOverlayPipeline(
        const std::string& vertGlsl, const std::string& fragGlsl, const char* name,
        const VkPipelineVertexInputStateCreateInfo& vertexInput,
        uint32_t pushConstantSize, bool alphaBlend,
        VkPipelineLayout& layoutOut, VkPipeline& pipelineOut,
        VkShaderStageFlags pushConstantStages = VK_SHADER_STAGE_VERTEX_BIT);

    // per frame
    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex, const FrameScene& scene);
    void setMonitorViewport(VkCommandBuffer cmd, int xOffset);
    void drawQuad(VkCommandBuffer cmd, TextureSlot slot, float x0, float y0, float x1, float y1);
    void renderFixationPoints(VkCommandBuffer cmd, const std::vector<NormalizedPoint2D>& points);
    void renderGazeDots(VkCommandBuffer cmd);
    void renderCalibrationDots(VkCommandBuffer cmd, const std::vector<NormalizedPoint2D>& points);
    void renderHeadGuide(VkCommandBuffer cmd, const HeadGuide& guide);

    // one anti-aliased shape on the CURRENT monitor viewport (shape pipeline must already be bound).
    // Centre / half size are in pixels of one monitor. outlinePx > 0 draws an outline of that thickness instead of a fill.
    void drawShape(VkCommandBuffer cmd, ShapeKind kind,
        float cxPx, float cyPx, float halfWPx, float halfHPx,
        const float* rgb, float alpha, float outlinePx = 0.0f);

    // texture helpers
    void uploadTextureData(Texture& tex, const void* pixels,
        VkFormat fmt, int w, int h);
    void destroyTexture(Texture& tex);
    void updateDescriptorSet(int slot, const Texture& tex);

    // vulkan helpers
    VkShaderModule createShaderModule(const std::vector<uint32_t>& spirv);

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem);
    uint32_t findMemoryType(uint32_t filter, VkMemoryPropertyFlags props);

    void transitionImageLayout(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout);

    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer cmd);

    // queries
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice dev);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice dev);

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& a);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& a);
    VkExtent2D chooseSwapExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& c);

    // validation
    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredExtensions();
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT*, void*);

private:
    // size of ONE monitor
    int m_monitorWidth = 0;
    int m_monitorHeight = 0;

    // config if in sdr/hdr mode
    bool m_isHDR = false;

    // core vulkan objs
    VkInstance m_instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    VkQueue m_presentQueue = VK_NULL_HANDLE;

    // swap chain
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> m_swapImages;
    std::vector<VkImageView> m_swapImageViews;
    VkFormat m_swapFormat = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR m_swapColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkExtent2D m_swapExtent{};

    // render pass/ frame buffers
    VkRenderPass m_renderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> m_framebuffers;

    // pipelines
    VkDescriptorSetLayout m_descSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;                 // textured quads

    VkPipelineLayout m_fixationPointLayout = VK_NULL_HANDLE;
    VkPipeline m_fixationPointPipeline = VK_NULL_HANDLE;    // cross

    VkPipelineLayout m_gazeDotLayout = VK_NULL_HANDLE;
    VkPipeline m_gazeDotPipeline = VK_NULL_HANDLE;          // instanced gaze dots

    VkPipelineLayout m_shapeLayout = VK_NULL_HANDLE;
    VkPipeline m_shapePipeline = VK_NULL_HANDLE;            // dots / ovals / rects

    // vertex buffer (fixation cross geometry)
    VkBuffer m_fixationPointVB = VK_NULL_HANDLE;
    VkDeviceMemory m_fixationPointVBMem = VK_NULL_HANDLE;

    // per-instance buffer for the gaze dots (host visible, persistently mapped)
    VkBuffer m_gazeDotVB = VK_NULL_HANDLE;
    VkDeviceMemory m_gazeDotVBMem = VK_NULL_HANDLE;
    void* m_gazeDotMapped = nullptr;
    uint32_t m_gazeDotCount = 0;

    // command pool
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> m_commandBuffers;

    static constexpr int MAX_FRAMES_IN_FLIGHT = 2; //allow CPU to prepare N + 1 frames while GPU is rendering N
    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> m_imageAvailable{};
    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> m_renderFinished{};
    std::array<VkFence, MAX_FRAMES_IN_FLIGHT> m_inFlightFences{};
    uint32_t m_currentFrame = 0;

    // textures and descriptors
    VkDescriptorPool m_descPool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, MAX_TEXTURES> m_descSets{};
    std::array<Texture, MAX_TEXTURES> m_textures{};

    std::array<DecodedImage, MAX_TEXTURES> m_decodedImages{};
};