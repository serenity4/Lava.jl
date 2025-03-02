# Mutability is to allow finalizers.
mutable struct BakedRenderGraph
  device::Device
  allocator::LinearAllocator
  index_data::IndexData
  nodes::Vector{RenderNode}
  resources::Dictionary{ResourceID, Resource}
  node_uses::Dictionary{NodeID,Dictionary{ResourceID, ResourceUsage}}
  combined_uses::Dictionary{ResourceID, ResourceUsage}
  # Pairs att1 => att2 where att1 is a multisampled attachment resolved on att2.
  resolve_pairs::Dictionary{Resource, Resource}
end

function combine_resource_uses(uses)
  combined_uses = Dictionary{ResourceID,ResourceUsage}()
  for node_uses in uses
    for (rid, resource_usage) in pairs(node_uses)
      existing = get(combined_uses, rid, nothing)
      if !isnothing(existing)
        combined_uses[rid] = combine(existing, resource_usage)
      else
        insert!(combined_uses, rid, resource_usage)
      end
    end
  end
  combined_uses
end

combine_resource_uses_per_node(uses) = dictionary(nid => dictionary(rid => reduce(merge, ruses) for (rid, ruses) in pairs(nuses)) for (nid, nuses) in pairs(uses))

function bake!(rg::RenderGraph)
  add_resource_dependencies!(rg)
  resolve_pairs = resolve_attachment_pairs(rg)
  add_resolve_attachments!(rg, resolve_pairs)

  node_uses = combine_resource_uses_per_node(rg.uses)

  # Materialize logical resources with properties derived from usage patterns.
  combined_uses = combine_resource_uses(node_uses)
  check_physical_resources(rg, combined_uses)
  materialized_resources = materialize_logical_resources(rg, combined_uses)
  allocate_blocks!(rg, materialized_resources)
  resources = dictionary(r.id => islogical(r) ? materialized_resources[r.id] : r for r in rg.resources)
  resolve_pairs = dictionary(resources[r.id] => resources[resolve_r.id] for (r, resolve_r) in pairs(resolve_pairs))

  # Allocate descriptors and get the batch index to free them when execution has finished.
  descriptors = descriptors_for_cycle(rg)
  descriptor_batch = write_descriptors!(rg.device.descriptors, descriptors, node_uses, resources)

  baked = BakedRenderGraph(rg.device, rg.allocator, IndexData(), sort_nodes(rg, node_uses), resources, node_uses, combined_uses, resolve_pairs)
  finalizer(x -> free_descriptor_batch!(rg.device.descriptors, descriptor_batch), baked)
end

function render!(rg::Union{RenderGraph,BakedRenderGraph})
  submission = SubmissionInfo(; signal_fence = get_fence!(rg.device))
  wait(render!(rg, submission))
end

function render!(rg::Union{RenderGraph,BakedRenderGraph}, submission::SubmissionInfo)
  command_buffer = request_command_buffer(rg.device)
  baked = render!(rg, command_buffer)
  push!(submission.free_after_completion, baked)
  submit!(submission, command_buffer)
end

render!(rg::RenderGraph, command_buffer::CommandBuffer) = render(command_buffer, bake!(rg))

render(device::Device, node::Union{RenderNode,Command}) = render(device, [node])
render(device::Device, nodes) = render!(RenderGraph(device, nodes))

function render(command_buffer::CommandBuffer, baked::BakedRenderGraph)
  records, pipeline_hashes = record_commands!(baked)
  create_pipelines!(baked.device)

  if any(!isempty(record.draws) for record in records)
    # Allocate index buffer.
    fill_indices!(baked.index_data, records)
    initialize_index_buffer(command_buffer, baked.device, baked.index_data)
  end
  # Fill command buffer with synchronization commands & recorded commands.
  flush(command_buffer, baked, records, pipeline_hashes)
  isa(command_buffer, SimpleCommandBuffer) && push!(command_buffer.to_free, baked)
  baked
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

function record_commands!(baked::BakedRenderGraph)
  records = CompactRecord[]
  pipeline_hashes = Dictionary{ProgramInstance,UInt64}()

  # Record commands and submit pipelines for creation.
  for node in baked.nodes
    record = CompactRecord(baked, node)
    push!(records, record)
    merge!(pipeline_hashes, request_pipelines(baked, record))
  end

  records, pipeline_hashes
end

function CompactRecord(baked::BakedRenderGraph, node::RenderNode)
  rec = CompactRecord(node, Dictionary(), Dictionary(), Command[], Command[])
  for command in node.commands
    record!(rec, command)
  end
  rec
end

function request_pipelines(baked::BakedRenderGraph, record::CompactRecord)
  (; device) = baked
  pipeline_hashes = Dictionary{ProgramInstance,UInt64}()
  layout = pipeline_layout(device)
  for (program, calls) in pairs(record.draws)
    for ((data, state), draws) in pairs(calls)
      for targets in unique!(last.(draws))
        info = pipeline_info_graphics(record.node.render_area::RenderArea, program, state.render_state, state.invocation_state, targets, layout, baked.resources)
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

function rendering_info(baked::BakedRenderGraph, node::RenderNode)
  color_attachments = Vk.RenderingAttachmentInfo[]
  depth_attachment = C_NULL
  stencil_attachment = C_NULL
  resolve_ids = Set{ResourceID}()
  uses = baked.node_uses[node.id]

  for use in uses
    (; id) = use
    in(id, resolve_ids) && continue
    resource = baked.resources[id]
    resource_type(resource) == RESOURCE_TYPE_ATTACHMENT || continue
    attachment_usage = use.usage::AttachmentUsage
    # Resolve attachments are grouped with their destination attachment.
    (; type) = attachment_usage
    (; attachment) = resource
    info = nothing
    for (usage, kind) in ((RESOURCE_USAGE_COLOR_ATTACHMENT, :color), (RESOURCE_USAGE_DEPTH_ATTACHMENT, :depth), (RESOURCE_USAGE_STENCIL_ATTACHMENT, :stencil))
      in(usage, type) || continue
      info = if attachment_usage.samples > 1
        resolve_resource = baked.resolve_pairs[resource]
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
