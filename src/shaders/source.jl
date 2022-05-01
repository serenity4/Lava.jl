"""
SPIR-V shader code, with stage and entry point information.
"""
@auto_hash_equals struct ShaderSource
  code::Vector{UInt8}
  stage::Vk.ShaderStageFlag
  entry_point::Symbol
end

function Base.show(io::IO, source::ShaderSource)
  print(io, "ShaderSource(", source.stage, ", ", length(source.code), " bytes)")
end

function ShaderSource(fname::AbstractString; stage = shader_stage(fname), entry_point = :main)
  ShaderSource(read(fname), stage, entry_point)
end

function ShaderSource(io::IO, stage::Vk.ShaderStageFlag, entry_point = :main)
  code = UInt8[]
  readbytes!(io, code, bytesavailable(io))
  ShaderSource(code, stage, entry_point)
end

function ShaderSource(f, argtypes, interface::ShaderInterface)
  cfg = CFG(f, argtypes, inferred = true)
  try
    ir = make_shader(cfg, interface)
    ret = validate_shader(ir)
    @assert !iserror(ret) unwrap_error(ret)
    ShaderSource(reinterpret(UInt8, assemble(ir)), shader_stage(interface.execution_model), :main)
  catch
    @error """
    Shader compilation failed. Showing inferred code:

    $(sprint(show, cfg.code; context = :color => true))
    """
    rethrow()
  end
end

macro shader(interface, ex)
  args = SPIRV.get_signature(ex)
  :(ShaderSource($(esc.(args)...), $(esc(interface))))
end
