#pragma once
#include <shaderc/shaderc.hpp>
#include <vector>
#include <stdexcept>
#include <string>

inline std::vector<uint32_t> compileGLSL(
    const std::string& src,
    shaderc_shader_kind kind,
    const char* name = "shader")
{
    shaderc::Compiler compiler;
    shaderc::CompileOptions opts;
    opts.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    opts.SetOptimizationLevel(shaderc_optimization_level_performance);

    auto result = compiler.CompileGlslToSpv(src, kind, name, opts);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success)
        throw std::runtime_error(std::string("Shader compile error: ")
            + result.GetErrorMessage());

    return { result.cbegin(), result.cend() };
}

// ---------------------------------------------------------------------------
// Pipeline 1: textured full-screen quad (instruction / fail / complete screens)
// ---------------------------------------------------------------------------

static const std::string QUAD_VERT_GLSL = R"glsl(
#version 450

layout(push_constant) uniform PC {
    float x0, y0, x1, y1;
    uint  texSlot;
} pc;

layout(location = 0) out vec2 vUV;

void main() {
    const vec2 corners[4] = vec2[](
        vec2(pc.x0, pc.y1),
        vec2(pc.x1, pc.y1),
        vec2(pc.x1, pc.y0),
        vec2(pc.x0, pc.y0)
    );
    // UV: Vulkan clip space Y+ is down; image was already flipped in CPU.
    // (0,0)=top-left of texture maps to top-left of quad.
    const vec2 uvs[4] = vec2[](
        vec2(0.0, 0.0),
        vec2(1.0, 0.0),
        vec2(1.0, 1.0),
        vec2(0.0, 1.0)
    );
    const int idx[6] = int[](0, 1, 2, 2, 3, 0);
    int i = idx[gl_VertexIndex];
    gl_Position = vec4(corners[i], 0.0, 1.0);
    vUV = uvs[i];
}
)glsl";

// this is the shader used for displaying the images
static const std::string QUAD_FRAG_GLSL = R"glsl(
#version 450

layout(set = 0, binding = 0) uniform sampler2D uTex;
layout(location = 0) in  vec2 vUV;
layout(location = 0) out vec4 outColor;

void main() {
    // no additional transformations required by the shader
    outColor = vec4(texture(uTex, vUV).rgb, 1.0);
}
)glsl";

// ---------------------------------------------------------------------------
// Pipeline 2: fixation cross (vertex buffer of 12 verts, scaled/translated by push constants)
// ---------------------------------------------------------------------------

static const std::string CROSSHAIR_VERT_GLSL = R"glsl(
#version 450

layout(location = 0) in vec2 aPos;

layout(push_constant) uniform PC {
    float centerX;
    float centerY;
    float halfSizeX;
    float halfSizeY;
} pc;

void main()
{
    gl_Position = vec4(
        aPos.x * pc.halfSizeX + pc.centerX,
        aPos.y * pc.halfSizeY + pc.centerY,
        0.0,
        1.0
    );
}
)glsl";

static const std::string CROSSHAIR_FRAG_GLSL = R"glsl(
#version 450

layout(location = 0) out vec4 outColor;

void main() {
    // just white
    outColor = vec4(1.0, 1.0, 1.0, 1.0);
}
)glsl";

// ---------------------------------------------------------------------------
// Pipeline 3: gaze dots (instanced; one instance per sample, quad generated in the vertex shader)
// Per-instance attributes: centre (NDC) + RGBA colour. Push constants: dot half size in NDC.
// ---------------------------------------------------------------------------

static const std::string GAZE_DOT_VERT_GLSL = R"glsl(
#version 450

layout(location = 0) in vec2 iCenter;
layout(location = 1) in vec4 iColor;

layout(push_constant) uniform PC {
    float halfSizeX;
    float halfSizeY;
} pc;

layout(location = 0) out vec2 vLocal;
layout(location = 1) out vec4 vColor;

void main() {
    const vec2 corners[4] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2( 1.0,  1.0),
        vec2(-1.0,  1.0)
    );
    const int idx[6] = int[](0, 1, 2, 2, 3, 0);
    vec2 local = corners[idx[gl_VertexIndex]];

    gl_Position = vec4(iCenter + local * vec2(pc.halfSizeX, pc.halfSizeY), 0.0, 1.0);
    vLocal = local;
    vColor = iColor;
}
)glsl";

static const std::string GAZE_DOT_FRAG_GLSL = R"glsl(
#version 450

layout(location = 0) in vec2 vLocal;
layout(location = 1) in vec4 vColor;
layout(location = 0) out vec4 outColor;

void main() {
    if (dot(vLocal, vLocal) > 1.0) discard; // outside the unit circle
    outColor = vColor;
}
)glsl";

// ---------------------------------------------------------------------------
// Pipeline 4: generic anti-aliased shapes (filled or outlined ellipse, filled rect).
// Used for the white calibration dots and the head positioning guide.
// One quad per shape, placed and coloured entirely by push constants (vertex + fragment stages).
// ix / iy are the inner radius of an outline as a fraction of the outer radius on each axis
// (0 = filled), so the outline can have the same pixel thickness on an oval's x and y axes.
// ---------------------------------------------------------------------------

static const std::string SHAPE_VERT_GLSL = R"glsl(
#version 450

layout(push_constant) uniform PC {
    float cx, cy;       // centre (NDC)
    float hx, hy;       // half size (NDC)
    float r, g, b, a;   // colour
    float ix, iy;       // outline inner radius / outer radius (0 = filled)
    float shape;        // 0 = ellipse, 1 = rect
} pc;

layout(location = 0) out vec2 vLocal;

void main() {
    const vec2 corners[4] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2( 1.0,  1.0),
        vec2(-1.0,  1.0)
    );
    const int idx[6] = int[](0, 1, 2, 2, 3, 0);
    vec2 local = corners[idx[gl_VertexIndex]];

    gl_Position = vec4(vec2(pc.cx, pc.cy) + local * vec2(pc.hx, pc.hy), 0.0, 1.0);
    vLocal = local;
}
)glsl";

static const std::string SHAPE_FRAG_GLSL = R"glsl(
#version 450

layout(push_constant) uniform PC {
    float cx, cy;
    float hx, hy;
    float r, g, b, a;
    float ix, iy;
    float shape;
} pc;

layout(location = 0) in  vec2 vLocal;
layout(location = 0) out vec4 outColor;

void main() {
    // distance from the centre, 1.0 at the outer edge (circle/ellipse) or the rectangle's border
    float d  = (pc.shape < 0.5) ? length(vLocal) : max(abs(vLocal.x), abs(vLocal.y));
    float aa = max(fwidth(d), 1e-4);                  // ~1 pixel wide edge
    float coverage = 1.0 - smoothstep(1.0 - aa, 1.0, d);

    if (pc.ix > 0.0 && pc.iy > 0.0) {                 // outline: cut out the inner ellipse
        float di  = length(vLocal / vec2(pc.ix, pc.iy));
        float aai = max(fwidth(di), 1e-4);
        coverage *= smoothstep(1.0 - aai, 1.0, di);
    }

    if (coverage <= 0.0) discard;
    outColor = vec4(pc.r, pc.g, pc.b, pc.a * coverage);
}
)glsl";
