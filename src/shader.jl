macro shader(model::QuoteNode, device, kwargs...)
  (ex, options, cached, interpreter, assemble) = parse_shader_kwargs(kwargs)
  propagate_source(__source__, esc(shader(device, ex, __module__, SPIRV.execution_models[model.value::Symbol], options, cached, interpreter, assemble)))
end

function parse_shader_kwargs(kwargs)
  ex = options = cached = interpreter = assemble = nothing
  for kwarg in kwargs
    @match kwarg begin
      Expr(:(=), :options, value) || :options && Do(value = :options) => (options = value)
      Expr(:(=), :interpreter, value) || :interpreter && Do(value = :interpreter) => (interpreter = value)
      Expr(:(=), :cached, value) || :cached && Do(value = :cached) => (cached = value)
      Expr(:(=), :assemble, value) || :assemble && Do(value = :assemble) => (assemble = value)
      Expr(:(=), parameter, value) => throw(ArgumentError("Received unknown parameter `$parameter` with value $value"))
      ::Expr => (ex = kwarg)
      _ => throw(ArgumentError("Expected parameter or expression as argument, got $kwarg"))
    end
  end
  !isnothing(ex) || throw(ArgumentError("Expected expression as positional argument"))
  (ex, options, cached, interpreter, assemble)
end

for (name, model) in pairs(SPIRV.execution_models)
  @eval macro $name(device, kwargs...)
    (ex, options, cached, interpreter, assemble) = parse_shader_kwargs(kwargs)
    propagate_source(__source__, esc(shader(device, ex, __module__, $model, options, cached, interpreter, assemble)))
  end
  @eval export $(Symbol("@$name"))
end

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

Vk.ShaderStageFlag(shader::Shader) = Vk.ShaderStageFlag(shader.info.interface.execution_model)

struct ShaderCache
  device::Vk.Device
  compilation_cache::ShaderCompilationCache
  shaders::Cache{IdDict{ShaderSource,Shader}}
  alignment::VulkanAlignment
end

function Base.empty!(cache::ShaderCache)
  empty!(cache.compilation_cache)
  empty!(cache.shaders)
  cache
end
ShaderCache(device, alignment) = ShaderCache(device, ShaderCompilationCache(), Cache{IdDict{ShaderSource,Shader}}(), alignment)

Shader(cache::ShaderCache, source::ShaderSource) = get!(cache, source)
Base.get!(cache::ShaderCache, source::ShaderSource) = get!(() -> Shader(cache.device, source), cache.shaders, source)

function shader(device, ex::Expr, __module__, execution_model, options, cached, interpreter, assemble)
  _device, _source, _cached, _compilation_cache, _interpreter, _assemble = gensym.((:device, :source, :cached, :compilation_cache, :interpreter, :assemble))
  quote
    $_device = $device
    isa($_device, $Device) || throw(ArgumentError("`Device` expected as first argument, got a value of type $(typeof(device))`"))
    $_assemble = something($assemble, true)
    $_cached = something($cached, true) & $_assemble
    $_interpreter = @something($interpreter, $SPIRVInterpreter())
    $_compilation_cache = $_cached ? $_device.shader_cache.compilation_cache : nothing
    $_source = $(SPIRV.shader(ex, __module__, execution_model, options, :($_device.spirv_features), :($_compilation_cache); assemble = :($_assemble), interpreter = :($_interpreter), layout = :(VulkanLayout($_device.shader_cache.alignment))))
    !$_assemble && return $_source
    $_cached ? $Shader($_device, $_source) : $Shader($_device.shader_cache.device, $_source)
  end
end

function Vk.PipelineShaderStageCreateInfo(shader::Shader)
  specialization_info = isempty(shader.specialization_constants) ? C_NULL : shader.specialization_constants
  Vk.PipelineShaderStageCreateInfo(Vk.ShaderStageFlag(shader), shader.shader_module, "main"; specialization_info)
end
