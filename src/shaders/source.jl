struct ShaderInfo
  mi::MethodInstance
  interface::ShaderInterface
  interp::SPIRVInterpreter
  layout::VulkanLayout
end

"""
SPIR-V shader code, with stage and entry point information.
"""
struct ShaderSource
  code::Vector{UInt8}
  info::ShaderInfo
end

Vk.ShaderStageFlag(source::ShaderSource) = shader_stage(source.info.interface.execution_model)

function Base.show(io::IO, source::ShaderSource)
  print(io, "ShaderSource(", source.info.interface.execution_model, ", ", length(source.code), " bytes)")
end

@struct_hash_equal struct ShaderSpec
  mi::MethodInstance
  interface::ShaderInterface
end
ShaderSpec(f, argtypes::Type, interface::ShaderInterface) = ShaderSpec(SPIRV.method_instance(f, argtypes), interface)

function ShaderSource(spec::ShaderSpec, alignment::VulkanAlignment)
  interp = SPIRVInterpreter()
  shader = SPIRV.Shader(spec.mi, spec.interface, interp, alignment)
  ret = validate(shader)
  !iserror(ret) || throw(unwrap_error(ret))
  ShaderSource(reinterpret(UInt8, assemble(shader)), ShaderInfo(spec.mi, spec.interface, interp, shader.layout))
end
