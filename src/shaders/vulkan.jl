"""
Vulkan shader, fed into the 
"""
struct Shader
  source::ShaderSource
  shader_module::Vk.ShaderModule
  push_constant_ranges::Vector{Vk.PushConstantRange}
  specialization_constants::Vector{Vk.SpecializationInfo}
end

device(shader::Shader) = shader.shader_module.device

Shader(source, shader_module) = Shader(source, shader_module, [], [])

struct ShaderCache
  device::Vk.Device
  shaders::Dictionary{ShaderSource,Shader}
end

Base.hash(source::ShaderSource, h::UInt64) = objectid(source.code) + h

ShaderCache(device) = ShaderCache(device, Dictionary())

Shader(cache::ShaderCache, source::ShaderSource) = find_shader!(cache, source)

function find_shader!(cache::ShaderCache, source::ShaderSource)
  if haskey(cache.shaders, source)
    cache.shaders[source]
  else
    shader_module = Vk.ShaderModule(cache.device, source)
    shader = Shader(source, shader_module)
    insert!(cache.shaders, source, shader)
    shader
  end
end

function Vk.ShaderModule(device, source::ShaderSource)
  length(source.code) % 4 == 0 || pad_shader_code!(source.code)
  Vk.ShaderModule(device, length(source.code), reinterpret(UInt32, source.code))
end

function pad_shader_code!(code::Vector{UInt8})
  size = cld(length(code), 4)
  rem = size * 4 - length(code)
  if rem ≠ 0
    resize!(code, size * 4)
    code[(end - rem + 1):end] .= 0
  end
  @assert length(code) % 4 == 0
  code
end

function Vk.PipelineShaderStageCreateInfo(shader::Shader)
  specialization_info = isempty(shader.specialization_constants) ? C_NULL : shader.specialization_constants
  Vk.PipelineShaderStageCreateInfo(shader.source.stage, shader.shader_module, string(shader.source.entry_point); specialization_info)
end
