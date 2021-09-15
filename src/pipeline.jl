# abstract type PipelineType end

# abstract type Compute <: PipelineType end
# abstract type Graphics <: PipelineType end
# abstract type RayTracing <: PipelineType end

@enum PipelineType::Int8 begin
    PIPELINE_TYPE_GRAPHICS
    PIPELINE_TYPE_COMPUTE
    PIPELINE_TYPE_ASYNC_COMPUTE
end

struct Pipeline
    handle::Vk.Pipeline
    type::PipelineType
end
