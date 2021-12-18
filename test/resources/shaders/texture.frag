#version 450

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require

layout(location = 0) out vec4 out_color;
layout(location = 0) in vec2 uv;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MaterialData {
    vec2 uv_scaling;
    uint sampler_id;
} md;

layout(set = 0, binding = 3) uniform sampler2D samplers[2048];

layout(push_constant) uniform DrawData {
    uint64_t camera; // unused
    uint64_t vbuffer; // unused
    uint64_t material;
} dd;

void main() {
    MaterialData md = MaterialData(dd.material);
    out_color = vec4(texture(samplers[md.sampler_id], uv * md.uv_scaling).rgb, 1.);
}
