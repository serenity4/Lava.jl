function Base.flush(cb::CommandBuffer, baked::BakedRenderGraph, records, pipeline_hashes)
  bind_state = BindState()
  sync_state = SynchronizationState()
  for record in records
    synchronize_before!(sync_state, cb, baked, record.node)
    bind_state = flush(cb, record, baked, bind_state, pipeline_hashes)
    synchronize_after!(sync_state, cb, baked, record.node)
  end
end

function Base.flush(cb::CommandBuffer, record::CompactRecord, baked::BakedRenderGraph, bind_state::BindState, pipeline_hashes)
  (; device) = baked

  begin_render_node(cb, baked, record.node)
  for (program, calls) in pairs(record.draws)
    for ((data, state), draws) in pairs(calls)
      for (draw, targets) in draws
        hash = pipeline_hashes[ProgramInstance(program, state, targets)]
        pipeline = device.pipeline_ht_graphics[hash]
        reqs = BindRequirements(pipeline, data, device.descriptors.gset)
        bind_state = bind(cb, reqs, bind_state)
        draw.type == COMMAND_TYPE_DRAW_INDEXED ? apply(cb, draw.impl::DrawIndexed, baked.index_data) : apply(cb, draw.impl::Union{DrawIndirect, DrawIndexedIndirect})
      end
    end
  end
  end_render_node(cb, baked, record.node)

  for (program, calls) in pairs(record.dispatches)
    hash = pipeline_hashes[ProgramInstance(program, nothing, nothing)]
    pipeline = device.pipeline_ht_compute[hash]
    for (data, dispatches) in pairs(calls)
      reqs = BindRequirements(pipeline, data, device.descriptors.gset)
      bind_state = bind(cb, reqs, bind_state)
      for dispatch in dispatches
        apply(cb, dispatch.impl::Union{Dispatch, DispatchIndirect})
      end
    end
  end

  bind_state
end

"""
Build barriers for all resources that require it.
"""
function synchronize_before!(state::SynchronizationState, cb, baked::BakedRenderGraph, node::RenderNode)
  info = dependency_info!(state, baked.node_uses, baked.resources, node)
  if !isempty(info.image_memory_barriers) || !isempty(info.buffer_memory_barriers)
    Vk.cmd_pipeline_barrier_2(cb, info)
  end
end

function begin_render_node(cb, baked::BakedRenderGraph, node::RenderNode)
  isnothing(node.render_area) && return
  Vk.cmd_begin_rendering(cb, rendering_info(baked, node))
end

function end_render_node(cb, baked::BakedRenderGraph, node::RenderNode)
  if !isnothing(node.render_area)
    Vk.cmd_end_rendering(cb)
  end
end

function synchronize_after!(state::SynchronizationState, cb, baked::BakedRenderGraph, node::RenderNode)
  nothing
end
