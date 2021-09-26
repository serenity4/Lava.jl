"""
Describes data that an object needs to be drawn, but without having a pipeline created yet.
"""
struct ShaderDependencies
    vertex_buffer::Buffer
    index_buffer::Optional{Buffer}
    descriptor_sets::Vector{DescriptorSet}
end
