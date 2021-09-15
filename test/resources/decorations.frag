#version 450

layout(set = 0, binding = 0) uniform writeonly image2D image;

layout(location = 0) in vec4 position;
layout(location = 0) out vec4 color;

void main() {
    color = position;
}
