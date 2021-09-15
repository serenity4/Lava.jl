struct ShaderSpecification
    source_file::String
    reuse_descriptors::Bool
    entry_point::Symbol
    stage::Vk.ShaderStageFlag
    language::ShaderLanguage
end

function ShaderSpecification(source_file, stage::Vk.ShaderStageFlag; reuse_descriptors = false, entry_point = :main)
    ShaderSpecification(source_file, reuse_descriptors, entry_point, stage, shader_language(source_file))
end

function ShaderSpecification(source_file, language::ShaderLanguage; reuse_descriptors = false, entry_point = :main)
    stage = @match language begin
        &GLSL || &HLSL => shader_stage(source_file, language)
        &SPIR_V => shader_stage(source_file, language) # will error, need to introspect into SPIR-V module
    end
    ShaderSpecification(source_file, reuse_descriptors, entry_point, stage, language)
end

function ShaderSpecification(source_file; reuse_descriptors = false, entry_point = :main)
    if shader_language(source_file) == SPIR_V
        ShaderSpecification(source_file, SPIR_V; reuse_descriptors, entry_point)
    else
        error("Language or stage must be supplied")
    end
end
