mutable struct Shader
  shader_module::Vk.ShaderModule
  info::ShaderInfo
  push_constant_ranges::Vector{Vk.PushConstantRange}
  specialization_constants::Vector{Vk.SpecializationInfo}
end

device(shader::Shader) = shader.shader_module.device

function Shader(device, source::ShaderSource)
  shader_module = Vk.ShaderModule(device, source)
  Shader(shader_module, source.info, Vk.PushConstantRange[], Vk.SpecializationInfo[])
end

shader_stage(shader::Shader) = shader_stage(shader.info.interface.execution_model)

struct ShaderCache
  device::Vk.Device
  # This could be reused across devices, if we want to avoid going through codegen again.
  # However the supported features would have to be taken as the intersection of all the supported
  # ones on each hypothetical device.
  compiled::Cache{Dict{ShaderSpec,ShaderSource}}
  shaders::Cache{IdDict{ShaderSource,Shader}}
end

function Base.empty!(cache::ShaderCache)
  empty!(cache.compiled)
  empty!(cache.shaders)
  cache
end

ShaderCache(device) = ShaderCache(device, Cache{Dict{ShaderSpec,ShaderSource}}(), Cache{IdDict{ShaderSource,Shader}}())

Shader(cache::ShaderCache, source::ShaderSource) = get!(cache, source)
ShaderSource(cache::ShaderCache, spec::ShaderSpec) = get!(cache, spec)

Base.get!(cache::ShaderCache, spec::ShaderSpec) = get!(() -> ShaderSource(spec), cache.compiled, spec)
Base.get!(cache::ShaderCache, source::ShaderSource) = get!(() -> Shader(cache.device, source), cache.shaders, source)

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
  Vk.PipelineShaderStageCreateInfo(shader_stage(shader), shader.shader_module, "main"; specialization_info)
end
