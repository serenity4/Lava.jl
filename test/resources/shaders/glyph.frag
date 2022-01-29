/*

Inputs:
- position
- curves
- text color
- pixel per em

Output: color

*/

#version 450

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require

layout(location = 0) in vec2 position;
// starting index in the curve buffer (i.e. character)
layout(location = 1) flat in uint start;
// number of curves to process
layout(location = 2) flat in uint curve_count;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MaterialData {
    float text_color[4];
    uint64_t curve_buffer_pointer;
    float pixel_per_em;
} md;

layout(push_constant) uniform DrawData {
    uint64_t camera; // unused
    uint64_t vbuffer; // unused
    uint64_t material;
} dd;

struct CurveData {
    vec2 p1;
    vec2 p2;
    vec2 p3;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer CurveBuffer {
    CurveData curves[];
} curve_buffer;

layout(location = 0) out vec4 out_color;

const float atol = 0.5e-6;

vec2 bezier_curve(vec2 p1, vec2 p2, vec2 p3, float t) {
    return (1 - t) * (1 - t) * p1 + 2 * t * (1 - t) * p2 + t * t * p3;
}

void main() {
    float intensity = 0.0;
    MaterialData md = MaterialData(dd.material);
    CurveBuffer curve_buffer = CurveBuffer(md.curve_buffer_pointer);

    for (uint curve_idx = start; curve_idx < start + curve_count; curve_idx++) {

        CurveData curve_points = curve_buffer.curves[curve_idx];

        vec2 p1 = curve_points.p1 - position;
        vec2 p2 = curve_points.p2 - position;
        vec2 p3 = curve_points.p3 - position;

        for (uint coord = 0; coord < 2; coord++) {

            float xbar_1 = p1[1 - coord];
            float xbar_2 = p2[1 - coord];
            float xbar_3 = p3[1 - coord];

            if (max(max(p1[coord], p2[coord]), p3[coord]) * md.pixel_per_em <= -0.5) continue;

            uint rshift = (xbar_1 > 0 ? 2 : 0) + (xbar_2 > 0 ? 4 : 0) + (xbar_3 > 0 ? 8 : 0);
            uint code = (0x2E74U >> rshift) & 3U;

            if (code != 0U) {
                float a = xbar_1 - 2 * xbar_2 + xbar_3;
                float b = xbar_1 - xbar_2;
                float c = xbar_1;

                float t1;
                float t2;

                if (abs(a) < atol) {
                    t1 = c / (2 * b);
                    t2 = t1;
                } else {
                    float Delta = b * b - a * c;
                    if (Delta < 0) continue;
                    float delta = sqrt(Delta);
                    t1 = (b - delta) / a;
                    t2 = (b + delta) / a;
                }

                float val = 0;

                if ((code & 1U) == 1U) {
                    val = clamp(md.pixel_per_em * bezier_curve(p1, p2, p3, t1)[coord] + 0.5, 0.0, 1.0);
                }
                if (code > 1U) {
                    val = -clamp(md.pixel_per_em * bezier_curve(p1, p2, p3, t2)[coord] + 0.5, 0.0, 1.0);
                }
                intensity += val * (coord == 1 ? 1 : -1);
            }
        }
    }
    intensity = clamp(intensity, -1, 1);
    intensity = sqrt(abs(intensity));
    float alpha = md.text_color[3] * intensity;
    out_color = vec4(md.text_color[0], md.text_color[1], md.text_color[2], alpha);
}
