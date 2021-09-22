struct PipelineBindingState
    pipeline::Pipeline
    push_ranges::Vector{Vk.PushConstantRange}
end

PipelineBindingState(pipeline) = PipelineBindingState(pipeline, [])

"""
Binding state that must be set in order for
drawing commands to render correctly.
"""
struct BindRequirements
    dependencies::ShaderDependencies
    pipeline_state::PipelineBindingState
    layout::PipelineLayout
    push_data::Any
end

BindRequirements(dependencies, pipeline_state, layout) = BindRequirements(dependencies, pipeline_state, layout, nothing)

"""
Describes the current binding state.
"""
struct BindState
    vertex_buffer::Optional{Buffer}
    index_buffer::Optional{Buffer}
    descriptor_sets::Vector{DescriptorSet}
    pipeline_state::Optional{PipelineBindingState}
    push_data::Any
end

BindState() = BindState(nothing, nothing, [], nothing, nothing)

function Base.bind(cbuffer::Vk.CommandBuffer, reqs::BindRequirements, state::BindState = BindState())
    (;vertex_buffer, index_buffer, descriptor_sets) = reqs.dependencies
    (;push_ranges, pipeline) = reqs.pipeline_state
    current_push_data = state.push_data
    layout = reqs.layout

    if pipeline ≠ state.pipeline_state.pipeline
        cmd_bind_pipeline(cbuffer, PIPELINE_BIND_POINT_GRAPHICS, pipeline)
        if push_ranges ≠ state.pipeline_state.push_ranges
            # push data gets invalidated
            current_push_data = nothing
        end
    end

    vertex_buffer ≠ state.vertex_buffer && cmd_bind_vertex_buffers(cbuffer, [vertex_buffer], [0])

    if !isnothing(index_buffer) && index_buffer ≠ state.index_buffer
        cmd_bind_index_buffer(cbuffer, index_buffer, 0, INDEX_TYPE_UINT32)
    end

    if !isempty(descriptor_sets)
        dsets = handle.(descriptor_sets)
        dsets ≠ state.descriptor_sets && cmd_bind_descriptor_sets(cbuffer, PIPELINE_BIND_POINT_GRAPHICS, layout, 0, dsets, [])
    end

    if !isnothing(push_data) && push_data ≠ current_push_data
        cmd_push_constants(cbuffer, layout, SHADER_STAGE_VERTEX_BIT, 1, Ref(push_data), sizeof(push_data))
    end

    BindState(vertex_buffer, index_buffer, descriptor_sets, PipelineBindingState(pipeline, push_ranges), push_data)
end
