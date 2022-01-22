function file_ext(stage::Vk.ShaderStageFlag)
    last_ext = @match stage begin
        &Vk.SHADER_STAGE_VERTEX_BIT => "vert"
        &Vk.SHADER_STAGE_FRAGMENT_BIT => "frag"
        &Vk.SHADER_STAGE_TESSELLATION_CONTROL_BIT => "tesc"
        &Vk.SHADER_STAGE_TESSELLATION_EVALUATION_BIT => "tese"
        &Vk.SHADER_STAGE_GEOMETRY_BIT => "geom"
        &Vk.SHADER_STAGE_COMPUTE_BIT => "comp"
        &Vk.SHADER_STAGE_RAYGEN_BIT_KHR => "rgen"
        &Vk.SHADER_STAGE_INTERSECTION_BIT_KHR => "rint"
        &Vk.SHADER_STAGE_ANY_HIT_BIT_KHR => "rahit"
        &Vk.SHADER_STAGE_CLOSEST_HIT_BIT_KHR => "rchit"
        &Vk.SHADER_STAGE_MISS_BIT_KHR => "rmiss"
        &Vk.SHADER_STAGE_CALLABLE_BIT_KHR => "rcall"
        &Vk.SHADER_STAGE_MESH_BIT_NV => "mesh"
        &Vk.SHADER_STAGE_TASK_BIT_NV => "task"
        _ => error("Unknown stage $stage")
    end
    string("spv.", last_ext)
end

"""
    shader_stage(file_ext)

Automatically detect a shader stage from a file extension.

## Examples

```jldoctest
julia> shader_stage("my_shader.frag.spv") == Vk.SHADER_STAGE_FRAGMENT_BIT
true

julia> shader_stage("my_shader.geom") == Vk.SHADER_STAGE_GEOMETRY_BIT
true
```
"""
function shader_stage(file::AbstractString)
    basename, file_ext = splitext(file)
    if file_ext == ".spv"
        # Maybe it is of the form .vert.spv?
        _, file_ext2 = splitext(basename)
        !isempty(file_ext2) && (file_ext = file_ext2)
    end
    @match file_ext begin
        ".vert" => Vk.SHADER_STAGE_VERTEX_BIT
        ".frag" => Vk.SHADER_STAGE_FRAGMENT_BIT
        ".tesc" => Vk.SHADER_STAGE_TESSELLATION_CONTROL_BIT
        ".tese" => Vk.SHADER_STAGE_TESSELLATION_EVALUATION_BIT
        ".geom" => Vk.SHADER_STAGE_GEOMETRY_BIT
        ".comp" => Vk.SHADER_STAGE_COMPUTE_BIT
        ".rgen" => Vk.SHADER_STAGE_RAYGEN_BIT_KHR
        ".rint" => Vk.SHADER_STAGE_INTERSECTION_BIT_KHR
        ".rahit" => Vk.SHADER_STAGE_ANY_HIT_BIT_KHR
        ".rchit" => Vk.SHADER_STAGE_CLOSEST_HIT_BIT_KHR
        ".rmiss" => Vk.SHADER_STAGE_MISS_BIT_KHR
        ".rcall" => Vk.SHADER_STAGE_CALLABLE_BIT_KHR
        ".mesh" => Vk.SHADER_STAGE_MESH_BIT_NV
        ".task" => Vk.SHADER_STAGE_TASK_BIT_NV
        _ => error("Unknown file extension $file_ext.
                    Expected a file with a common shader extension name, like .vert or .frag,
                    possibly suffixed with .spv like .vert.spv or .frag.spv")
    end
end

function shader_stage(execution_model::SPIRV.ExecutionModel)
    @match execution_model begin
        &SPIRV.ExecutionModelVertex                 => Vk.SHADER_STAGE_VERTEX_BIT
        &SPIRV.ExecutionModelTessellationControl    => Vk.SHADER_STAGE_TESSELLATION_CONTROL_BIT
        &SPIRV.ExecutionModelTessellationEvaluation => Vk.SHADER_STAGE_TESSELLATION_EVALUATION_BIT
        &SPIRV.ExecutionModelGeometry               => Vk.SHADER_STAGE_GEOMETRY_BIT
        &SPIRV.ExecutionModelFragment               => Vk.SHADER_STAGE_FRAGMENT_BIT
        &SPIRV.ExecutionModelGLCompute              => Vk.SHADER_STAGE_COMPUTE_BIT
        &SPIRV.ExecutionModelTaskNV                 => Vk.SHADER_STAGE_TASK_BIT_NV
        &SPIRV.ExecutionModelMeshNV                 => Vk.SHADER_STAGE_MESH_BIT_NV
        &SPIRV.ExecutionModelRayGenerationKHR       => Vk.SHADER_STAGE_RAYGEN_BIT_KHR
        &SPIRV.ExecutionModelIntersectionKHR        => Vk.SHADER_STAGE_INTERSECTION_BIT_KHR
        &SPIRV.ExecutionModelAnyHitKHR              => Vk.SHADER_STAGE_ANY_HIT_BIT_KHR
        &SPIRV.ExecutionModelClosestHitKHR          => Vk.SHADER_STAGE_CLOSEST_HIT_BIT_KHR
        &SPIRV.ExecutionModelMissKHR                => Vk.SHADER_STAGE_MISS_BIT_KHR
        &SPIRV.ExecutionModelCallableKHR            => Vk.SHADER_STAGE_CALLABLE_BIT_KHR
    end
end
