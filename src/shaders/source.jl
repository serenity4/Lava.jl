struct ShaderSource
    code::Vector{UInt8}
    language::ShaderLanguage
    stage::Vk.ShaderStageFlag
    entry_points::Vector{Symbol}
end

function Base.show(io::IO, source::ShaderSource)
    print(io, "ShaderSource(", source.language, ", ", source.stage, ", ", length(source.code), " bytes)")
end

function pad_shader_code!(code::Vector{UInt8})
    size = cld(length(code), 4)
    rem = size * 4 - length(code)
    if rem ≠ 0
        resize!(code, size * 4)
        code[end - rem + 1:end] .= 0
    end
    @assert length(code) % 4 == 0
    code
end

struct Shader
    source::ShaderSource
    shader_module::Vk.ShaderModule
    entry_point::Symbol
    push_constant_ranges::Vector{Vk.PushConstantRange}
    specialization_constants::Vector{Vk.SpecializationInfo}
end

device(shader::Shader) = shader.shader_module.device

Shader(source, shader_module, entry_point) = Shader(source, shader_module, entry_point, [], [])

struct ShaderCache
    device::Vk.Device
    compiled::Dictionary{String,ShaderSource}
    shaders::Dictionary{ShaderSource,Shader}
end

Base.hash(source::ShaderSource, h::UInt64) = objectid(source.code) + h

ShaderCache(device) = ShaderCache(device, Dictionary(), Dictionary())

function find_source!(cache::ShaderCache, spec::ShaderSpecification)
    file = string(spec.source_file)
    if haskey(cache.compiled, file)
        source = cache.compiled[file]
        if spec.entry_point in source.entry_points
            return source
        end
    else
        source = ShaderSource(spec)
        if source.language ≠ SPIR_V
            source = compile(source)
        end
        insert!(cache.compiled, file, source)
        source
    end
end

function ShaderSource(spec::ShaderSpecification)
    ShaderSource(read(spec.source_file), spec.language, spec.stage, [spec.entry_point])
end

function find_shader!(cache::ShaderCache, source::ShaderSource, entry_point::Symbol)
    if source.language ≠ SPIR_V
        source = compile(source)
    end
    find_shader!(cache, source, ShaderSpecification("", false, entry_point, SPIR_V))
end

find_shader!(cache::ShaderCache, spec::ShaderSpecification) = find_shader!(cache, find_source!(cache, spec), spec)

function find_shader!(cache::ShaderCache, source::ShaderSource, spec::ShaderSpecification)
    if haskey(cache.shaders, source)
        cache.shaders[source]
    else
        shader_module = Vk.ShaderModule(cache.device, source)
        shader = Shader(source, shader_module, spec.entry_point)
        insert!(cache.shaders, source, shader)
        shader
    end
end

"""
Retrieve a shader from the provided cache.

Note that the cache will be modified if it does not contain the requested shader.
"""
Shader(cache::ShaderCache, spec::ShaderSpecification) = find_shader!(cache, spec)
