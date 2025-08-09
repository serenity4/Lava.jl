function Base.flush(command_buffer::CommandBuffer, rg::RenderGraph, records, pipeline_hashes)
  bind_state = BindState()
  sync_state = SynchronizationState()
  for record in records
    @debug "Flushing node $(sprint(print_name, record.node))"
    synchronize_before!(sync_state, command_buffer, rg, record.node)
    bind_state = flush(command_buffer, record, rg, bind_state, pipeline_hashes)
    synchronize_after!(sync_state, command_buffer, rg, record.node)
  end
end

function Base.flush(command_buffer::CommandBuffer, record::CompactRecord, rg::RenderGraph, bind_state::BindState, pipeline_hashes)
  (; device) = rg
  (; node) = record

  for (resource, clear) in pairs(node.clears)
    isimage(resource) || continue
    resource = get_physical_resource(rg, resource)
    (; image) = resource
    match_subresource(image) do aspect, layer_range, mip_range, layout
      subresource = Subresource(aspect, layer_range, mip_range)
      new_layout = Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
      transition_layout(command_buffer, image, subresource, layout, new_layout)
      range = Vk.ImageSubresourceRange(subresource)
      is_color = isa(clear.data, NTuple{4})
      if is_color
        clear = Vk.ClearColorValue(clear.data)
        Vk.cmd_clear_color_image(command_buffer, image, new_layout, clear, [range])
      else
        clear = Vk.ClearDepthStencilValue(clear)
        Vk.cmd_clear_depth_stencil_image(command_buffer, image, new_layout, clear, [range])
      end
    end
  end

  begin_render_node(command_buffer, rg, node)
  attachment_clears = nothing
  rects = nothing
  for (resource, clear) in pairs(node.clears)
    isattachment(resource) || continue
    resource = get_physical_resource(rg, resource)
    (; attachment) = resource
    attachment_clears === nothing && (attachment_clears = Vk.ClearAttachment[])
    rects === nothing && (rects = Vk.ClearRect[])
    is_color = isa(clear.data, NTuple{4})
    aspect = is_color ? Vk.IMAGE_ASPECT_COLOR_BIT : Vk.IMAGE_ASPECT_DEPTH_BIT | Vk.IMAGE_ASPECT_STENCIL_BIT
    push!(attachment_clears, Vk.ClearAttachment(aspect, 0, Vk.ClearValue(clear)))
    range = layer_range(attachment)
    push!(rects, Vk.ClearRect(node.render_area.rect, range.start - 1, 1 + range.stop - range.start))
  end
  if attachment_clears !== nothing
    @assert length(attachment_clears) == length(rects)
    Vk.cmd_clear_attachments(command_buffer, attachment_clears, rects)
  end
  for (program, calls) in pairs(record.draws)
    for ((data, state), commands) in pairs(calls)
      for (command, targets) in commands
        hash = pipeline_hashes[ProgramInstance(program, state, targets)]
        pipeline = device.pipeline_ht_graphics[hash]
        reqs = BindRequirements(pipeline, data, device.descriptors.gset, state.render_state)
        bind_state = bind(command_buffer, reqs, bind_state)
        command.type == COMMAND_TYPE_DRAW_INDEXED ? apply(command_buffer, command.graphics.draw::DrawIndexed, rg.index_data) : apply(command_buffer, command.graphics.draw::Union{DrawIndirect, DrawIndexedIndirect}, rg.materialized_resources)
      end
    end
  end
  end_render_node(command_buffer, rg, node)

  for (program, calls) in pairs(record.dispatches)
    hash = pipeline_hashes[ProgramInstance(program, nothing, nothing)]
    pipeline = device.pipeline_ht_compute[hash]
    for (data, commands) in pairs(calls)
      reqs = BindRequirements(pipeline, data, device.descriptors.gset)
      bind_state = bind(command_buffer, reqs, bind_state)
      for command in commands
        command.type == COMMAND_TYPE_DISPATCH ? apply(command_buffer, command.compute.dispatch::Dispatch) : apply(command_buffer, command.compute.dispatch::DispatchIndirect, rg.materialized_resources)
      end
    end
  end

  for command in record.transfers
    apply(command_buffer, command.transfer, rg.materialized_resources)
  end

  bind_state
end

"""
Build barriers for all resources that require it.
"""
function synchronize_before!(state::SynchronizationState, command_buffer::CommandBuffer, rg::RenderGraph, node::RenderNode)
  dependency = dependency_info!(state, rg, node)
  dependency === nothing && return
  Vk._cmd_pipeline_barrier_2(command_buffer, dependency)
end

function begin_render_node(command_buffer::CommandBuffer, rg::RenderGraph, node::RenderNode)
  isnothing(node.render_area) && return
  Vk.cmd_begin_rendering(command_buffer, rendering_info(rg, node))
end

function end_render_node(command_buffer::CommandBuffer, rg::RenderGraph, node::RenderNode)
  if !isnothing(node.render_area)
    Vk.cmd_end_rendering(command_buffer)
  end
end

function synchronize_after!(state::SynchronizationState, command_buffer::CommandBuffer, rg::RenderGraph, node::RenderNode)
  nothing
end
