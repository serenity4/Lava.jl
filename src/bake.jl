# Mutability is to allow finalizers.
mutable struct BakedRenderGraph
  device::Device
  allocator::LinearAllocator
  index_data::IndexData
  nodes::Vector{RenderNode}
  resources::PhysicalResources
  uses::Dictionary{NodeUUID,ResourceUses}
  # Pairs att1 => att2 where att1 is a multisampled attachment resolved on att2.
  resolve_pairs::Dictionary{ResourceUUID, ResourceUUID}
end

function bake!(device::Device, rg::RenderGraph)
  generate_draw_infos!(rg)
  resolve_pairs = resolve_attachment_pairs(rg)
  add_resolve_attachments(rg, resolve_pairs)
  uses = ResourceUses(rg)
  check_physical_resources(rg, uses)
  resources = merge(materialize_logical_resources(rg, uses), rg.physical_resources)
  descriptors = materialize_logical_descriptors!(rg.device, resources, rg.uses)
  baked = BakedRenderGraph(rg.device, rg.allocator, IndexData(), sort_nodes(rg), resources, rg.uses, uuid.(resolve_pairs))
  finalizer(x -> free_logical_descriptors!(x.device, descriptors), baked)
end

function render(rg::Union{RenderGraph,BakedRenderGraph})
  command_buffer = request_command_buffer(rg.device)
  baked = render(command_buffer, rg)
  wait(submit(command_buffer, SubmissionInfo(signal_fence = fence(rg.device), release_after_completion = [baked])))
end

render(command_buffer::CommandBuffer, rg::RenderGraph) = render(command_buffer, bake!(rg.device, rg))

function render(command_buffer::CommandBuffer, baked::BakedRenderGraph)
  records, pipeline_hashes = record_commands!(baked)
  create_pipelines(baked.device)

  # Fill command buffer with synchronization commands & recorded commands.
  fill_indices!(baked.index_data, records)
  initialize(command_buffer, baked.device, baked.index_data)
  flush(command_buffer, baked, records, pipeline_hashes)
  baked
end

function fill_indices!(index_data::IndexData, records)
  for record in records
    for draws in record.programs
      for calls in draws
        for (command, target) in calls
          if isa(command, DrawIndexed)
            append!(index_data, command)
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

function Base.flush(cb::CommandBuffer, baked::BakedRenderGraph, records, pipeline_hashes)
  binding_state = BindState()
  state = SynchronizationState()
  for (node, record) in zip(baked.nodes, records)
    synchronize_before!(state, cb, baked, node)
    begin_render_node(cb, baked, node)
    binding_state = flush(cb, record, baked.device, binding_state, pipeline_hashes, baked.device.descriptors, baked.index_data)
    end_render_node(cb, baked, node)
    synchronize_after!(state, cb, baked, node)
  end
end

function rendering_info(rg::BakedRenderGraph, node::RenderNode)
  color_attachments = Vk.RenderingAttachmentInfo[]
  depth_attachment = C_NULL
  stencil_attachment = C_NULL
  resolve_uuids = Set{ResourceUUID}()

  (; attachments) = rg.resources
  attachment_uses = rg.uses[node.uuid].attachments

  for (uuid, attachment_usage) in pairs(attachment_uses)
    # Resolve attachments are grouped with their destination attachment.
    uuid in resolve_uuids && continue

    attachment = attachments[uuid]
    (; aspect) = attachment_usage
    info = if is_multisampled(attachment_usage)
      resolve_uuid = rg.resolve_pairs[uuid]
      push!(resolve_uuids, resolve_uuid)
      rendering_info(attachment, attachment_usage, attachments[resolve_uuid], attachment_uses[resolve_uuid])
    else
      rendering_info(attachment, attachment_usage)
    end
    if Vk.IMAGE_ASPECT_COLOR_BIT in aspect
      push!(color_attachments, info)
    elseif Vk.IMAGE_ASPECT_DEPTH_BIT in aspect
      depth_attachment == C_NULL || error("Multiple depth attachments detected (node: $(node.uuid))")
      depth_attachment = info
    elseif Vk.IMAGE_ASPECT_STENCIL_BIT in aspect
      stencil_attachment == C_NULL || error("Multiple stencil attachments detected (node: $(node.uuid))")
      stencil_attachment = info
    else
      error("Attachment is not a depth, color or stencil attachment as per its aspect value $aspect (node: $(node.uuid))")
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
SyncRequirements(usage::ResourceUsage) = SyncRequirements(access_bits(usage), usage.stages)

struct ResourceState
  sync_reqs::SyncRequirements
  last_accesses::Dictionary{Vk.AccessFlag2,Vk.PipelineStageFlag2}
  current_layout::RefValue{Vk.ImageLayout}
end

ResourceState(usage::BufferUsage) = ResourceState(SyncRequirements(usage), Dictionary(), Ref{Vk.ImageLayout}())
ResourceState(usage::ResourceUsage, layout) = ResourceState(SyncRequirements(usage), Dictionary(), layout)
ResourceState(layout::Ref{Vk.ImageLayout} = Ref{Vk.ImageLayout}()) = ResourceState(SyncRequirements(), Dictionary(), layout)

struct SynchronizationState
  resources::Dictionary{ResourceUUID,ResourceState}
end

SynchronizationState() = SynchronizationState(Dictionary())

must_synchronize(sync_reqs::SyncRequirements, usage) = WRITE in usage.access || iszero(sync_reqs.stages)
must_synchronize(sync_reqs::SyncRequirements, usage, from_layout, to_layout) = must_synchronize(sync_reqs, usage) || from_layout ≠ to_layout

function synchronize!(state::SynchronizationState, resource::PhysicalBuffer, usage::BufferUsage)
  rstate = get!(ResourceState, state.resources, resource.uuid)
  sync_reqs = restrict_sync_requirements(rstate.last_accesses, SyncRequirements(usage))
  must_synchronize(sync_reqs, usage) || return
  WRITE in usage.access && (state.resources[resource.uuid] = ResourceState(usage))
  Vk.BufferMemoryBarrier2(
    0,
    0,
    resource.buffer,
    resource.offset,
    resource.size;
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

function synchronize!(state::SynchronizationState, resource::PhysicalResource, usage::ResourceUsage)
  rstate = get!(() -> ResourceState(resource.layout), state.resources, resource.uuid)
  sync_reqs = restrict_sync_requirements(rstate.last_accesses, rstate.sync_reqs)
  from_layout = rstate.current_layout[]
  to_layout = image_layout(usage)
  must_synchronize(sync_reqs, usage, from_layout, to_layout) || return
  rstate.current_layout[] = to_layout
  WRITE in usage.access && (state.resources[resource.uuid] = ResourceState(usage, resource.layout))
  Vk.ImageMemoryBarrier2(
    from_layout,
    to_layout,
    0,
    0,
    resource.image,
    subresource_range(resource);
    src_access_mask = rstate.sync_reqs.access,
    dst_access_mask = sync_reqs.access,
    src_stage_mask = rstate.sync_reqs.stages,
    dst_stage_mask = sync_reqs.stages,
  )
end

add_barrier!(info::Vk.DependencyInfo, barrier::Vk.BufferMemoryBarrier2) = push!(info.buffer_memory_barriers, barrier)
add_barrier!(info::Vk.DependencyInfo, barrier::Vk.ImageMemoryBarrier2) = push!(info.image_memory_barriers, barrier)

function dependency_info!(state::SynchronizationState, baked::BakedRenderGraph, node::RenderNode)
  info = Vk.DependencyInfo([], [], [])
  uses = baked.uses[node.uuid]
  for (resource_uuid, usage) in pairs(uses.buffers)
    barrier = synchronize!(state, baked.resources.buffers[resource_uuid], usage)
    !isnothing(barrier) && add_barrier!(info, barrier)
  end
  for (resource_uuid, usage) in pairs(uses.images)
    barrier = synchronize!(state, baked.resources.images[resource_uuid], usage)
    !isnothing(barrier) && add_barrier!(info, barrier)
  end
  for (resource_uuid, usage) in pairs(uses.attachments)
    barrier = synchronize!(state, baked.resources.attachments[resource_uuid], usage)
    !isnothing(barrier) && add_barrier!(info, barrier)
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
