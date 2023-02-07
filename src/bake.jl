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
  generate_command_infos!(rg, materialized_resources)
  resources = dictionary(r.id => islogical(r) ? materialized_resources[r.id] : r for r in rg.resources)
  resolve_pairs = dictionary(resources[r.id] => resources[resolve_r.id] for (r, resolve_r) in pairs(resolve_pairs))

  descriptors = write_descriptors!(rg.device.descriptors, node_uses, resources)
  baked = BakedRenderGraph(rg.device, rg.allocator, IndexData(), sort_nodes(rg, node_uses), resources, node_uses, combined_uses, resolve_pairs)
  finalizer(x -> free_unused_descriptors!(x.device.descriptors), baked)
end

function render!(rg::Union{RenderGraph,BakedRenderGraph})
  command_buffer = request_command_buffer(rg.device)
  baked = render!(rg, command_buffer)
  wait(submit(command_buffer, SubmissionInfo(signal_fence = fence(rg.device), free_after_completion = [baked])))
end

render!(rg::RenderGraph, command_buffer::CommandBuffer) = render(command_buffer, bake!(rg))

function render(device::Device, node::RenderNode)
  rg = RenderGraph(device)
  add_node!(rg, node)
  render!(rg)
end

function render(device::Device, nodes)
  rg = RenderGraph(device)
  for node in nodes
    add_node!(rg, node)
  end
  render!(rg)
end

function render(command_buffer::CommandBuffer, baked::BakedRenderGraph)
  records, pipeline_hashes = record_commands!(baked)
  create_pipelines!(baked.device)

  if any(!isempty(record.draws) for record in records)
    # Allocate index buffer.
    fill_indices!(baked.index_data, records)
    initialize(command_buffer, baked.device, baked.index_data)
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
            append!(index_data, command.impl::DrawIndexed)
          end
        end
      end
    end
  end
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
  rec = CompactRecord(node, Dictionary(), Dictionary())
  for info in node.command_infos
    if is_graphics_command(info.command)
      @reset info.targets = materialize(baked, info.targets)
      draw!(rec, info)
    else
      dispatch!(rec, info)
    end
  end
  rec
end

function request_pipelines(baked::BakedRenderGraph, record::CompactRecord)
  (; device) = baked
  pipeline_hashes = Dictionary{ProgramInstance,UInt64}()
  for (program, calls) in pairs(record.draws)
    for ((data, state), draws) in pairs(calls)
      for targets in unique!(last.(draws))
        info = pipeline_info_graphics(device, record.node.render_area::RenderArea, program, state.render_state, state.invocation_state, targets)
        hash = request_pipeline(device, info)
        set!(pipeline_hashes, ProgramInstance(program, state, targets), hash)
      end
    end
  end
  for (program, calls) in pairs(record.dispatches)
    info = pipeline_info_compute(device, program)
    hash = request_pipeline(device, info)
    set!(pipeline_hashes, ProgramInstance(program, nothing, nothing), hash)
  end
  pipeline_hashes
end

function materialize(baked::BakedRenderGraph, targets::RenderTargets)
  color = map(targets.color) do resource
    islogical(resource) ? baked.resources[resource.id] : resource
  end
  depth = isnothing(targets.depth) ? nothing : islogical(targets.depth) ? baked.resources[targets.depth.id] : targets.depth
  stencil = isnothing(targets.stencil) ? nothing : islogical(targets.stencil) ? baked.resources[targets.stencil.id] : targets.stencil
  RenderTargets(color, depth, stencil)
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
    attachment = resource.data::Attachment
    (; aspect) = attachment_usage
    info = if attachment_usage.samples > 1
      resolve_resource = baked.resolve_pairs[resource]
      push!(resolve_ids, resolve_resource.id)
      rendering_info(attachment, attachment_usage, resolve_resource.data::Attachment, uses[resolve_resource.id].usage::AttachmentUsage)
    else
      rendering_info(attachment, attachment_usage)
    end
    if Vk.IMAGE_ASPECT_COLOR_BIT in aspect
      push!(color_attachments, info)
    elseif Vk.IMAGE_ASPECT_DEPTH_BIT in aspect
      depth_attachment == C_NULL || error("Multiple depth attachments detected (node: $(node.id))")
      depth_attachment = info
    elseif Vk.IMAGE_ASPECT_STENCIL_BIT in aspect
      stencil_attachment == C_NULL || error("Multiple stencil attachments detected (node: $(node.id))")
      stencil_attachment = info
    else
      error("Attachment is not a depth, color or stencil attachment as per its aspect value $aspect (node: $(node.id))")
    end
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
