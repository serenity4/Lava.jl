"""
Binding state that must be set in order for
drawing commands to render correctly.
"""
struct BindRequirements
  pipeline::Pipeline
  push_data::Any
  "Global bind-once descriptor set. May require rebinding when pipeline layouts or pipeline types change."
  unique_dset::DescriptorSet
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
  (; pipeline, push_data) = reqs
  current_push_data = state.push_data

  if pipeline ≠ state.pipeline
    bind_point = Vk.PipelineBindPoint(pipeline.type)
    Vk.cmd_bind_pipeline(cbuffer, bind_point, pipeline)
    if isnothing(state.pipeline) || state.pipeline.layout ≠ pipeline.layout || state.pipeline.type ≠ pipeline.type
      Vk.cmd_bind_descriptor_sets(cbuffer, bind_point, pipeline.layout, 0, [reqs.unique_dset], [])
    end
    if isnothing(state.pipeline) || pipeline.layout.push_constant_ranges ≠ state.pipeline.layout.push_constant_ranges
      # push data gets invalidated
      current_push_data = nothing
    end
  end

  if !isnothing(push_data) && push_data ≠ current_push_data
    ref = Ref(push_data)
    GC.@preserve ref Vk.cmd_push_constants(cbuffer, pipeline.layout, Vk.SHADER_STAGE_ALL, 0, sizeof(push_data), Base.unsafe_convert(Ptr{Cvoid}, ref))
  end
end
