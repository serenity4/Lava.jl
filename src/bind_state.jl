"""
Binding state that must be set in order for
drawing commands to render correctly.
"""
Base.@kwdef struct BindRequirements
  pipeline::Pipeline
  push_data::Any
  "Global bind-once descriptor set. May require rebinding when pipeline layouts or pipeline types change."
  unique_dset::DescriptorSet
  # Dynamic states.
  # Must not be `nothing` for graphics pipelines.
  enable_depth_testing::Optional{Bool} = nothing
  enable_depth_write::Optional{Bool} = nothing
  depth_compare_op::Optional{Vk.CompareOp} = nothing
  enable_stencil_testing::Optional{Bool} = nothing
  stencil_front::Optional{Vk.StencilOpState} = nothing
  stencil_back::Optional{Vk.StencilOpState} = nothing
end

BindRequirements(pipeline, push_data, unique_dset) = BindRequirements(; pipeline, push_data, unique_dset)

BindRequirements(pipeline, push_data, unique_dset, state::RenderState) = BindRequirements(; pipeline, push_data, unique_dset, state.enable_depth_testing, state.enable_depth_write, state.depth_compare_op, state.enable_stencil_testing, state.stencil_front, state.stencil_back)

"""
Describes the current binding state.
"""
struct BindState
  pipeline::Optional{Pipeline}
  push_data::Any
end

BindState() = BindState(nothing, nothing)

function Base.bind(command_buffer::CommandBuffer, reqs::BindRequirements, state::BindState = BindState())
  (; pipeline, push_data) = reqs
  current_push_data = state.push_data

  if pipeline ≠ state.pipeline
    bind_point = Vk.PipelineBindPoint(pipeline.type)
    Vk.cmd_bind_pipeline(command_buffer, bind_point, pipeline)
    if isnothing(state.pipeline) || state.pipeline.layout ≠ pipeline.layout || state.pipeline.type ≠ pipeline.type
      Vk.cmd_bind_descriptor_sets(command_buffer, bind_point, pipeline.layout, 0, [reqs.unique_dset], [])
    end
    if isnothing(state.pipeline) || pipeline.layout.push_constant_ranges ≠ state.pipeline.layout.push_constant_ranges
      # push data gets invalidated
      current_push_data = nothing
    end
  end

  if !isnothing(push_data) && push_data ≠ current_push_data
    ref = Ref(push_data)
    GC.@preserve ref Vk.cmd_push_constants(command_buffer, pipeline.layout, Vk.SHADER_STAGE_ALL, 0, sizeof(push_data), Base.unsafe_convert(Ptr{Cvoid}, ref))
  end

  if pipeline.type == PIPELINE_TYPE_GRAPHICS
    Vk.cmd_set_depth_test_enable(command_buffer, reqs.enable_depth_testing)
    Vk.cmd_set_depth_write_enable(command_buffer, reqs.enable_depth_write)
    Vk.cmd_set_depth_compare_op(command_buffer, reqs.depth_compare_op)

    Vk.cmd_set_stencil_test_enable(command_buffer, reqs.enable_stencil_testing)

    front = reqs.stencil_front
    Vk.cmd_set_stencil_op(command_buffer, Vk.STENCIL_FACE_FRONT_BIT, front.fail_op, front.pass_op, front.depth_fail_op, front.compare_op)
    Vk.cmd_set_stencil_compare_mask(command_buffer, Vk.STENCIL_FACE_FRONT_BIT, front.compare_mask)
    Vk.cmd_set_stencil_write_mask(command_buffer, Vk.STENCIL_FACE_FRONT_BIT, front.write_mask)
    Vk.cmd_set_stencil_reference(command_buffer, Vk.STENCIL_FACE_FRONT_BIT, front.reference)

    back = reqs.stencil_front
    Vk.cmd_set_stencil_op(command_buffer, Vk.STENCIL_FACE_BACK_BIT, back.fail_op, back.pass_op, back.depth_fail_op, back.compare_op)
    Vk.cmd_set_stencil_compare_mask(command_buffer, Vk.STENCIL_FACE_BACK_BIT, back.compare_mask)
    Vk.cmd_set_stencil_write_mask(command_buffer, Vk.STENCIL_FACE_BACK_BIT, back.write_mask)
    Vk.cmd_set_stencil_reference(command_buffer, Vk.STENCIL_FACE_BACK_BIT, back.reference)
  end

  BindState(pipeline, push_data)
end
