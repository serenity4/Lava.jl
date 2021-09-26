# abstract type PipelineType end

# abstract type Compute <: PipelineType end
# abstract type Graphics <: PipelineType end
# abstract type RayTracing <: PipelineType end

@enum PipelineType::Int8 begin
    PIPELINE_TYPE_GRAPHICS
    PIPELINE_TYPE_COMPUTE
    PIPELINE_TYPE_ASYNC_COMPUTE
end

struct PipelineLayout <: LavaAbstraction
    handle::Vk.PipelineLayout
    descriptor_set_layouts::Vector{Vk.DescriptorSetLayout}
    push_constant_ranges::Vector{Vk.PushConstantRange}
end

struct Pipeline <: LavaAbstraction
    handle::Vk.Pipeline
    type::PipelineType
    layout::PipelineLayout
end
