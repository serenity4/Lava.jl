"""
Computation unit that uses shaders as part of a graphics or compute pipeline.

It exposes a program interface through its shader interfaces and its shader resources.
"""
@auto_hash_equals struct Program
  shaders::Dictionary{Vk.ShaderStageFlag,Shader}
end

function Program(cache::ShaderCache, shaders...)
  shaders = map(shaders) do shader
    shader.stage => find_shader!(cache, shader)
  end
  Program(dictionary(shaders))
end

function Program(device, shaders...)
  Program(device.shader_cache, shaders...)
end

@auto_hash_equals struct RenderTargets
  color::Vector{PhysicalAttachment}
  depth::Optional{PhysicalAttachment}
  stencil::Optional{PhysicalAttachment}
end

RenderTargets(color; depth = nothing, stencil = nothing) = RenderTargets(color, depth, stencil)

"""
Program to be compiled into a pipeline with a specific state.
"""
@auto_hash_equals struct ProgramInstance
  program::Program
  state::DrawState
  targets::RenderTargets
end
