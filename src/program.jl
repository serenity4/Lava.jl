"""
Computation unit that uses shaders as part of a graphics or compute pipeline.

It exposes a program interface through its shader interfaces and its shader resources.
"""
struct Program
    input_type::Type
    shaders
    descriptor_sets
    push_constants
    specialization_constants
    attachments
end

"""
Set of data required to call a program.
"""
struct ProgramInterface
    vbuffer::Buffer
    ibuffer::Optional{Buffer}
    descriptors::Vector{DescriptorSet}
    push_constants::Dictionary{Vk.PushConstantRange,Any}
end

struct ProgramInvocation
    interface::ProgramInterface
    draw_source::DrawSource
end
