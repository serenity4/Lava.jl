struct PipelineBindingState
    pipeline::Pipeline
    layout::PipelineLayout
    push_ranges::Vector{Vk.PushConstantRange}
end

PipelineBindingState(pipeline) = PipelineBindingState(pipeline, [])

"""
Binding state that must be set in order for
drawing commands to render correctly.
"""
struct BindRequirements
    pipeline::Pipeline
    push_data::Any
end

"""
Describes the current binding state.
"""
struct BindState
    pipeline::Optional{Pipeline}
    push_data::Any
end

BindState() = BindState(nothing, nothing)

function Base.bind(cbuffer::Vk.CommandBuffer, reqs::BindRequirements, state::BindState = BindState())
    (;pipeline, push_data) = reqs
    current_push_data = state.push_data

    if pipeline ≠ state.pipeline
        cmd_bind_pipeline(cbuffer, PIPELINE_BIND_POINT_GRAPHICS, pipeline)
        if pipeline.layout.push_constant_ranges ≠ state.pipeline.layout.push_constant_ranges
            # push data gets invalidated
            current_push_data = nothing
        end
    end

    if !isnothing(push_data) && push_data ≠ current_push_data
        cmd_push_constants(cbuffer, pipeline.layout, SHADER_STAGE_VERTEX_BIT, 1, Ref(push_data), sizeof(push_data))
    end
end
