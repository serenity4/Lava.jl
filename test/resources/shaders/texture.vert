#version 450

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require

struct VertexData {
    vec2 pos;
    vec2 uv;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertexBuffer {
    VertexData data[];
};

layout(push_constant) uniform DrawData {
    uint64_t camera; // unused
    uint64_t vbuffer;
    uint64_t material; // unused
} dd;

layout(location = 0) out vec2 uv;

void main() {
    VertexData vd = VertexBuffer(dd.vbuffer).data[gl_VertexIndex];
    gl_Position = vec4(vd.pos, 0.0, 1.0);
    uv = vd.uv;
}
