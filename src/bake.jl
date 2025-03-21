function combine_resource_uses!(rg::RenderGraph)
  for node_uses in rg.combined_node_uses
    for (rid, resource_usage) in pairs(node_uses)
      existing = get(rg.combined_resource_uses, rid, nothing)
      if !isnothing(existing)
        rg.combined_resource_uses[rid] = combine(existing, resource_usage)
      else
        insert!(rg.combined_resource_uses, rid, resource_usage)
      end
    end
  end
end

function combine_resource_uses_per_node!(rg::RenderGraph)
  for (nid, nuses) in pairs(rg.uses)
    uses = dictionary(rid => reduce(merge, ruses) for (rid, ruses) in pairs(nuses))
    insert!(rg.combined_node_uses, nid, uses)
  end
end

function bake!(rg::RenderGraph)
  add_resource_dependencies!(rg)
  resolve_attachment_pairs!(rg)
  add_resolve_attachments!(rg)
  combine_resource_uses_per_node!(rg)

  # Materialize logical resources with properties derived from usage patterns.
  combine_resource_uses!(rg)
  check_physical_resources!(rg)
  materialize_logical_resources!(rg)
  allocate_blocks!(rg)

  # Allocate descriptors and get the batch index to free them when execution has finished.
  descriptors = descriptors_for_cycle(rg)
  write_descriptors!(rg, descriptors)
  rg
end

function render!(rg::RenderGraph; submission::Optional{SubmissionInfo} = nothing, wait = true)
  submission = @something(submission, sync_submission(rg.device))
  command_buffer = request_command_buffer(rg.device)
  render!(rg, command_buffer)
  push!(submission.free_after_completion)
  execution = submit!(submission, command_buffer)
  !wait && return execution
  @__MODULE__().wait(execution)
end

function render!(rg::RenderGraph, command_buffer::CommandBuffer)
  bake!(rg)
  render(command_buffer, rg)
end

render(device::Device, node::Union{RenderNode,Command}) = render(device, [node])
function render(device::Device, nodes)
  rg = RenderGraph(device, nodes)
  ret = render!(rg)
  finish!(rg)
  ret
end

function render(command_buffer::CommandBuffer, rg::RenderGraph)
  records, pipeline_hashes = record_commands!(rg)
  create_pipelines!(rg.device)

  if any(!isempty(record.draws) for record in records)
    # Allocate index buffer.
    fill_indices!(rg.index_data, records)
    initialize_index_buffer(command_buffer, rg.device, rg.index_data)
  end
  # Fill command buffer with synchronization commands & recorded commands.
  flush(command_buffer, rg, records, pipeline_hashes)
  isa(command_buffer, SimpleCommandBuffer) && push!(command_buffer.to_free, rg)
  rg
end

function fill_indices!(index_data::IndexData, records)
  for record in records
    for draws in record.draws
      for calls in draws
        for (command, target) in calls
          if command.type == COMMAND_TYPE_DRAW_INDEXED
            append!(index_data, command.graphics.draw::DrawIndexed)
          end
        end
      end
    end
  end
end

"""
Program to be compiled into a pipeline with a specific state.
"""
@struct_hash_equal struct ProgramInstance
  program::Program
  state::Optional{DrawState}
  targets::Optional{RenderTargets}
end

function record_commands!(rg::RenderGraph)
  records = CompactRecord[]
  pipeline_hashes = Dictionary{ProgramInstance,UInt64}()

  # Record commands and submit pipelines for creation.
  for node in rg.nodes
    record = CompactRecord(rg, node)
    push!(records, record)
    merge!(pipeline_hashes, request_pipelines(rg, record))
  end

  records, pipeline_hashes
end

function CompactRecord(rg::RenderGraph, node::RenderNode)
  rec = CompactRecord(node, Dictionary(), Dictionary(), Command[], Command[])
  for command in node.commands
    record!(rec, command)
  end
  rec
end

function request_pipelines(rg::RenderGraph, record::CompactRecord)
  (; device) = rg
  pipeline_hashes = Dictionary{ProgramInstance,UInt64}()
  layout = pipeline_layout(device)
  for (program, calls) in pairs(record.draws)
    for ((data, state), draws) in pairs(calls)
      for targets in unique!(last.(draws))
        info = pipeline_info_graphics(record.node.render_area::RenderArea, program, state.render_state, state.invocation_state, targets, layout, rg.materialized_resources)
        hash = request_pipeline(device, info)
        set!(pipeline_hashes, ProgramInstance(program, state, targets), hash)
      end
    end
  end
  for (program, calls) in pairs(record.dispatches)
    info = pipeline_info_compute(program, layout)
    hash = request_pipeline(device, info)
    set!(pipeline_hashes, ProgramInstance(program, nothing, nothing), hash)
  end
  pipeline_hashes
end

function rendering_info(rg::RenderGraph, node::RenderNode)
  color_attachments = Vk.RenderingAttachmentInfo[]
  depth_attachment = C_NULL
  stencil_attachment = C_NULL
  resolve_ids = Set{ResourceID}()
  uses = rg.combined_node_uses[node.id]

  for use in uses
    (; id) = use
    in(id, resolve_ids) && continue
    resource = get_physical_resource(rg, id)
    resource_type(resource) == RESOURCE_TYPE_ATTACHMENT || continue
    attachment_usage = use.usage::AttachmentUsage
    # Resolve attachments are grouped with their destination attachment.
    (; type) = attachment_usage
    (; attachment) = resource
    info = nothing
    for (usage, kind) in ((RESOURCE_USAGE_COLOR_ATTACHMENT, :color), (RESOURCE_USAGE_DEPTH_ATTACHMENT, :depth), (RESOURCE_USAGE_STENCIL_ATTACHMENT, :stencil))
      in(usage, type) || continue
      info = if attachment_usage.samples > 1
        resolve_resource = get_physical_resource(rg, rg.resolve_pairs[resource])
        push!(resolve_ids, resolve_resource.id)
        rendering_info(attachment, attachment_usage, kind, resolve_resource.attachment, uses[resolve_resource.id].usage::AttachmentUsage)
      else
        rendering_info(attachment, attachment_usage, kind)
      end
      @match kind begin
        :color => push!(color_attachments, info)
        :depth => begin
          depth_attachment == C_NULL || error("Multiple depth attachments detected (node: $(node.id))")
          depth_attachment = info
        end
        :stencil => begin
          stencil_attachment == C_NULL || error("Multiple stencil attachments detected (node: $(node.id))")
          stencil_attachment = info
        end
      end
    end
    isnothing(info) && error("Attachment is not a depth, color or stencil attachment as per its resource usage: $type (node: $(node.id))")
  end
  info = Vk.RenderingInfo(
    node.render_area.rect,
    1,
    0,
    color_attachments;
    depth_attachment,
    stencil_attachment,
  )
end
