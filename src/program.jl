"""
Computation unit that uses shaders as part of a graphics or compute pipeline.

It exposes a program interface through its shader interfaces and its shader resources.
"""
@auto_hash_equals struct Program
    shaders::Dictionary{Vk.ShaderStageFlag, Shader}
end

function Program(cache::ShaderCache, shaders::ShaderSpecification...)
    shaders = map(shaders) do shader_spec
        shader_spec.stage => find_shader!(cache, shader_spec)
    end
    Program(dictionary(shaders))
end

function Program(device, shaders::ShaderSpecification...)
    Program(device.shader_cache, shaders...)
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

"""
Program to be compiled into a pipeline with a specific state.
"""
@auto_hash_equals struct ProgramInstance
    program::Program
    state::DrawState
    targets::TargetAttachments
end
