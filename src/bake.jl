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
  expand_program_invocations!(rg)
  resolve_pairs = resolve_attachment_pairs(rg)
  add_resolve_attachments!(rg, resolve_pairs)

  node_uses = combine_resource_uses_per_node(rg.uses)

  # Materialize logical resources with properties derived from usage patterns.
  combined_uses = combine_resource_uses(node_uses)
  check_physical_resources(rg, combined_uses)
  materialized_resources = materialize_logical_resources(rg, combined_uses)
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

function Base.flush(cb::CommandBuffer, baked::BakedRenderGraph, records, pipeline_hashes)
  binding_state = BindState()
  state = SynchronizationState()
  for (node, record) in zip(baked.nodes, records)
    synchronize_before!(state, cb, baked, node)
    begin_render_node(cb, baked, node)
    binding_state = flush(cb, record, baked.device, binding_state, pipeline_hashes, baked.index_data)
    end_render_node(cb, baked, node)
    synchronize_after!(state, cb, baked, node)
  end
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

struct SyncRequirements
  access::Vk.AccessFlag2
  stages::Vk.PipelineStageFlag2
end

SyncRequirements() = SyncRequirements(0, 0)
SyncRequirements(usage) = SyncRequirements(access_flags(usage.type, usage.access, usage.stages), usage.stages)

struct ResourceState
  sync_reqs::SyncRequirements
  last_accesses::Dictionary{Vk.AccessFlag2,Vk.PipelineStageFlag2}
  current_layout::RefValue{Vk.ImageLayout} # for images and attachments only
end

ResourceState(usage::BufferUsage) = ResourceState(SyncRequirements(usage), Dictionary(), Ref{Vk.ImageLayout}())
ResourceState(usage, layout) = ResourceState(SyncRequirements(usage), Dictionary(), layout)
ResourceState(layout::Ref{Vk.ImageLayout} = Ref{Vk.ImageLayout}()) = ResourceState(SyncRequirements(), Dictionary(), layout)

struct SynchronizationState
  resources::Dictionary{ResourceID,ResourceState}
end

SynchronizationState() = SynchronizationState(Dictionary())

must_synchronize(sync_reqs::SyncRequirements, usage) = WRITE in usage.access || iszero(sync_reqs.stages)
must_synchronize(sync_reqs::SyncRequirements, usage, from_layout, to_layout) = must_synchronize(sync_reqs, usage) || from_layout ≠ to_layout

function synchronize_buffer_access!(state::SynchronizationState, resource::Resource, usage::BufferUsage)
  rstate = get!(ResourceState, state.resources, resource.id)
  sync_reqs = restrict_sync_requirements(rstate.last_accesses, SyncRequirements(usage))
  must_synchronize(sync_reqs, usage) || return
  WRITE in usage.access && (state.resources[resource.id] = ResourceState(usage))
  buffer = resource.data::Buffer
  Vk.BufferMemoryBarrier2(
    0,
    0,
    buffer,
    buffer.offset,
    buffer.size;
    src_access_mask = rstate.sync_reqs.access,
    dst_access_mask = sync_reqs.access,
    src_stage_mask = rstate.sync_reqs.stages,
    dst_stage_mask = sync_reqs.stages,
  )
end

function restrict_sync_requirements(last_accesses, sync_reqs)
  remaining_sync_stages = sync_reqs.stages
  for (access, stages) in pairs(last_accesses)
    if covers(access, sync_reqs.access)
      remaining_sync_stages &= ~stages
    end
    iszero(remaining_sync_stages) && break
  end
  @set sync_reqs.stages = remaining_sync_stages
end

function synchronize_image_access!(state::SynchronizationState, resource::Resource, usage::Union{ImageUsage, AttachmentUsage})
  image = @match usage begin
    ::ImageUsage => resource.data::Image
    ::AttachmentUsage => (resource.data::Attachment).view.image
  end
  rstate = get!(() -> ResourceState(image.layout), state.resources, resource.id)
  sync_reqs = restrict_sync_requirements(rstate.last_accesses, rstate.sync_reqs)
  from_layout = rstate.current_layout[]
  to_layout = image_layout(usage.type, usage.access)
  must_synchronize(sync_reqs, usage, from_layout, to_layout) || return
  rstate.current_layout[] = to_layout
  WRITE in usage.access && (state.resources[resource.id] = ResourceState(usage, rstate.current_layout))
  subresource = @match usage begin
    ::ImageUsage => subresource_range(image)
    ::AttachmentUsage => subresource_range((resource.data::Attachment).view)
  end
  Vk.ImageMemoryBarrier2(
    from_layout,
    to_layout,
    0,
    0,
    image.handle,
    subresource;
    src_access_mask = rstate.sync_reqs.access,
    dst_access_mask = sync_reqs.access,
    src_stage_mask = rstate.sync_reqs.stages,
    dst_stage_mask = sync_reqs.stages,
  )
end

function dependency_info!(state::SynchronizationState, baked::BakedRenderGraph, node::RenderNode)
  info = Vk.DependencyInfo([], [], [])
  uses = baked.node_uses[node.id]
  for use in uses
    resource = baked.resources[use.id]
    @switch resource_type(resource) begin
      @case &RESOURCE_TYPE_BUFFER
      barrier = synchronize_buffer_access!(state, resource, use.usage::BufferUsage)
      !isnothing(barrier) && push!(info.buffer_memory_barriers, barrier)

      @case &RESOURCE_TYPE_IMAGE
      barrier = synchronize_image_access!(state, resource, use.usage::ImageUsage)
      !isnothing(barrier) && push!(info.image_memory_barriers, barrier)

      @case &RESOURCE_TYPE_ATTACHMENT
      barrier = synchronize_image_access!(state, resource, use.usage::AttachmentUsage)
      !isnothing(barrier) && push!(info.image_memory_barriers, barrier)
    end
  end
  info
end

"""
Build barriers for all resources that require it.
"""
function synchronize_before!(state::SynchronizationState, cb, baked::BakedRenderGraph, node::RenderNode)
  info = dependency_info!(state, baked, node)
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


"""
Return whether `x` covers accesses of type `y`; that is, whether guarantees about memory access operations in `x` induce guarantees about memory access operations in `y`.

```jldoctest
julia> covers(Vk.ACCESS_MEMORY_WRITE_BIT, Vk.ACCESS_SHADER_WRITE_BIT)
true

julia> covers(Vk.ACCESS_SHADER_READ_BIT, Vk.ACCESS_UNIFORM_READ_BIT)
true

julia> covers(Vk.ACCESS_SHADER_WRITE_BIT, Vk.ACCESS_MEMORY_WRITE_BIT)
false
```

"""
function covers(x::Vk.AccessFlag, y::Vk.AccessFlag)
  y in x && return true
  if Vk.ACCESS_MEMORY_READ_BIT in x
    if Vk.ACCESS_MEMORY_READ_BIT ∉ y
      x &= ~Vk.ACCESS_MEMORY_READ_BIT
    end
    x |= |(
      Vk.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
      Vk.ACCESS_COLOR_ATTACHMENT_READ_BIT,
      Vk.ACCESS_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT,
      Vk.ACCESS_COMMAND_PREPROCESS_READ_BIT_NV,
      Vk.ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT,
      Vk.ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
      Vk.ACCESS_FRAGMENT_DENSITY_MAP_READ_BIT_EXT,
      Vk.ACCESS_HOST_READ_BIT,
      Vk.ACCESS_INDEX_READ_BIT,
      Vk.ACCESS_INDIRECT_COMMAND_READ_BIT,
      Vk.ACCESS_INPUT_ATTACHMENT_READ_BIT,
      Vk.ACCESS_SHADER_READ_BIT,
      Vk.ACCESS_SHADING_RATE_IMAGE_READ_BIT_NV,
      Vk.ACCESS_TRANSFER_READ_BIT,
      Vk.ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT,
      Vk.ACCESS_UNIFORM_READ_BIT,
      Vk.ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
    )
    y in x && return true
  end
  if Vk.ACCESS_MEMORY_WRITE_BIT in x
    if Vk.ACCESS_MEMORY_WRITE_BIT ∉ y
      x &= ~Vk.ACCESS_MEMORY_WRITE_BIT
    end
    x |= |(
      Vk.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
      Vk.ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      Vk.ACCESS_COMMAND_PREPROCESS_WRITE_BIT_NV,
      Vk.ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
      Vk.ACCESS_HOST_WRITE_BIT,
      Vk.ACCESS_SHADER_WRITE_BIT,
      Vk.ACCESS_TRANSFER_WRITE_BIT,
      Vk.ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
      Vk.ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
    )
    y in x && return true
  end
  if Vk.ACCESS_SHADER_READ_BIT in x
    if Vk.ACCESS_SHADER_READ_BIT ∉ y
      x &= ~Vk.ACCESS_SHADER_READ_BIT
    end
    x |= Vk.ACCESS_UNIFORM_READ_BIT
    y in x && return true
  end
  false
end
