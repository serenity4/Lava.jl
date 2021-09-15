struct ShaderSource
    code::Vector{UInt8}
    language::ShaderLanguage
    stage::Vk.ShaderStageFlag
    entry_points::Vector{Symbol}
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
    descriptor_infos::Vector{DescriptorInfo}
end

struct ShaderCache
    device::Device
    compiled::Dictionary{String,ShaderSource}
    shaders::Dictionary{ShaderSource,Shader}
end

Base.hash(source::ShaderSource, h::UInt64) = objectid(source.code) + h

ShaderCache(device) = ShaderCache(device, Dictionary{String,ShaderSource}(), Dictionary{ShaderSource,Shader}())

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
        shader_module = ShaderModule(cache.device, source)
        ir = IR(SPIRV.Module(IOBuffer(source.code)))
        infos = descriptor_infos(ir)
        shader = Shader(source, shader_module, spec.entry_point, infos)
        insert!(cache.shaders, source, shader)
        shader
    end
end

function descriptor_infos(ir::IR)
    filter!(!isnothing, map(pairs(ir.global_vars)) do (id, var)
        decs = var.decorations
        # only select variables with a descriptor set and binding assigned
        haskey(decs, SPIRV.DecorationDescriptorSet) && haskey(decs, SPIRV.DecorationBinding) || return nothing

        type = @match s = var.storage_class begin
            &SPIRV.StorageClassStorageBuffer => Vk.DESCRIPTOR_TYPE_STORAGE_BUFFER
            &SPIRV.StorageClassImage => @match var.type begin
                ::ImageType => Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE
            end
            &SPIRV.StorageClassUniformConstant => @match var.type begin
                ::SamplerType => Vk.DESCRIPTOR_TYPE_SAMPLER
                ::SampledImageType => Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE
                ::ImageType => Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE
                if haskey(decs, SPIRV.DecorationBlock) end => Vk.DESCRIPTOR_TYPE_UNIFORM_BUFFER
                _ => error(var)
            end
            &SPIRV.StorageClassUniform => @match var.type begin
                if haskey(decs, SPIRV.DecorationBufferBlock) end => Vk.DESCRIPTOR_TYPE_STORAGE_BUFFER
                _ => Vk.DESCRIPTOR_TYPE_UNIFORM_BUFFER
            end
            _ => error("Could not map variable $var to a Vulkan descriptor type.")
        end

        DescriptorInfo(type, decs[SPIRV.DecorationDescriptorSet][], decs[SPIRV.DecorationBinding][])
    end)
end
