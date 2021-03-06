#version 450

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require

struct VertexData {
    vec2 position;
    // starting index in the curve buffer (i.e. character)
    uint start;
    // number of curves to process
    uint curve_count;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertexBuffer {
    VertexData data[];
};

layout(push_constant) uniform DrawData {
    uint64_t camera; // unused
    uint64_t vbuffer;
    uint64_t material; // unused
} dd;

layout(location = 0) out vec2 position;
layout(location = 1) out uint start;
layout(location = 2) out uint curve_count;

void main() {
    VertexData vd = VertexBuffer(dd.vbuffer).data[gl_VertexIndex];
    gl_Position = vec4(vd.position.xy, 0.0, 1.0);
    position = vd.position;
    start = vd.start;
    curve_count = vd.curve_count;
}
