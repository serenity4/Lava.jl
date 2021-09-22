struct ShaderCompilationError <: Exception
    msg
end

Base.showerror(io::IO, err::ShaderCompilationError) = print(io, "ShaderCompilationError:\n\n$(err.msg)")

"""
    compile(shader)

Compile a shader file in text format to SPIR-V.
"""
function compile(source::ShaderSource; extra_flags=[], validate_spirv=true)::ShaderSource
    flags = ["-V"; "--stdin"; extra_flags]
    source.language == SPIR_V && error("Shader $source already compiled")
    source.language == HLSL && push!(flags, "-D")
    if validate_spirv && "--spirv-val" ∉ flags
        push!(flags, "--spirv-val")
    end

    if isempty(source.entry_points)
        entry_point = :main
    elseif length(source.entry_points) == 1
        entry_point = first(source.entry_points)
    else
        error("Multiple entry points not supported in uncompiled source")
    end

    dst = string(tempname(), ".spv")
    input = IOBuffer()
    write(input, source.code)
    seekstart(input)
    err = IOBuffer()

    append!(flags, [
        "-e",
        string(entry_point),
        "-S",
        string(file_ext(source.language, source.stage)),
        "-o",
        dst,
    ])
    try
        run(pipeline(`$glslangValidator $flags`, stdin=input, stdout=err))
    catch e
        if e isa ProcessFailedException
            err_str = String(take!(err))
            throw(ShaderCompilationError(err_str))
        else
            rethrow(e)
        end
    end

    code = read(dst)
    rm(dst)
    ShaderSource(code, SPIR_V, source.stage, [entry_point])
end

function Vk.ShaderModule(device, source::ShaderSource)
    length(source.code) % 4 == 0 || pad_shader_code!(source.code)
    Vk.ShaderModule(device, length(source.code), reinterpret(UInt32, source.code))
end

function Vk.PipelineShaderStageCreateInfo(shader::Shader; specialization_info = C_NULL)
    Vk.PipelineShaderStageCreateInfo(shader.source.stage, shader.shader_module, string(shader.entry_point); specialization_info)
end
