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

for (name, model) in pairs(SPIRV.EXECUTION_MODELS)
  @eval macro $name(device, kwargs...)
    cached = true
    kwargs = collect(kwargs)
    for (i, kwarg) in enumerate(kwargs)
      Meta.isexpr(kwarg, :(=), 2) && kwarg.args[1] === :cached && (cached = kwarg.args[2]; deleteat!(kwargs, i))
    end
    (ex, options, features, cache, assemble, layout, interpreter) = SPIRV.parse_shader_kwargs(kwargs)
    _device, _features, _layout, _cache, _assemble, _source = gensym.([:device, :features, :layout, :cache, :assemble, :source])
    compile_ex = SPIRV.compile_shader_ex(ex, __module__, $model; options, features = _features, cache = _cache, interpreter, layout = _layout, assemble = _assemble)
    propagate_source(__source__, esc(quote
      $_device = $device
      isa($_device, $Device) || error("`Device` expected as first positional argument, got value of type ", typeof($_device))
      $_features = something($features, $_device.spirv_features)
      $_layout = something($layout, $VulkanLayout($_device.alignment))
      $_cache = $cached === true ? something($cache, $_device.shader_cache.compilation_cache) : nothing
      $_assemble = something($assemble, true)
      $_source = $compile_ex
      $cached === true ? $Shader($_device.shader_cache, $_source) : $Shader($_device.handle, $_source)
    end))
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
end

function Base.empty!(cache::ShaderCache)
  empty!(cache.compilation_cache)
  empty!(cache.shaders)
  cache
end
ShaderCache(device) = ShaderCache(device, ShaderCompilationCache(), Cache{IdDict{ShaderSource,Shader}}())

Shader(cache::ShaderCache, source::ShaderSource) = get!(cache, source)
Base.get!(cache::ShaderCache, source::ShaderSource) = get!(() -> Shader(cache.device, source), cache.shaders, source)

function Vk.PipelineShaderStageCreateInfo(shader::Shader)
  specialization_info = isempty(shader.specialization_constants) ? C_NULL : shader.specialization_constants
  Vk.PipelineShaderStageCreateInfo(Vk.ShaderStageFlag(shader), shader.shader_module, "main"; specialization_info)
end
