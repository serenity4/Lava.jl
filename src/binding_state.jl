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

function Base.bind(cbuffer::CommandBuffer, reqs::BindRequirements, state::BindState = BindState())
    (;pipeline, push_data) = reqs
    current_push_data = state.push_data

    if pipeline ≠ state.pipeline
        Vk.cmd_bind_pipeline(cbuffer, Vk.PipelineBindPoint(pipeline.type), pipeline)
        if isnothing(state.pipeline) || pipeline.layout.push_constant_ranges ≠ state.pipeline.layout.push_constant_ranges
            # push data gets invalidated
            current_push_data = nothing
        end
    end

    if !isnothing(push_data) && push_data ≠ current_push_data
        ref = Ref(push_data)
        GC.@preserve ref Vk.cmd_push_constants(cbuffer, pipeline.layout, Vk.ShaderStageFlag(0x7fffffff), 0, sizeof(push_data), Base.unsafe_convert(Ptr{Cvoid}, ref))
    end
end
