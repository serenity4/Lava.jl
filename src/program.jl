"""
Computation unit that uses shaders as part of a graphics or compute pipeline.

It exposes a program interface through its shader interfaces and its shader resources.
"""
struct Program
    input_type::Type
    shaders::Dictionary{Vk.ShaderStageFlag, Shader}
end

function Program(@nospecialize(T::Type), cache, shaders::ShaderSpecification...)
    shaders = map(shaders) do shader_spec
        shader_spec.stage => find_shader!(cache, shader_spec)
    end
    Program(T, dictionary(shaders))
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
