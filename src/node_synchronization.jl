struct SyncRequirements
  access::Vk.AccessFlag2
  stages::Vk.PipelineStageFlag2
end

SyncRequirements() = SyncRequirements(0, 0)
SyncRequirements(usage) = SyncRequirements(access_flags(usage.type, usage.access, usage.stages), usage.stages)

struct BufferResourceState
  sync::SyncRequirements
  accesses::Dictionary{Vk.AccessFlag2,Vk.PipelineStageFlag2}
end

BufferResourceState(usage::BufferUsage) = BufferResourceState(SyncRequirements(usage), Dictionary{Vk.AccessFlag2,Vk.PipelineStageFlag2}())
BufferResourceState() = BufferResourceState(SyncRequirements(), Dictionary{Vk.AccessFlag2,Vk.PipelineStageFlag2}())

struct SubresourceState
  sync::SyncRequirements
  accesses::Dictionary{Vk.AccessFlag2,Vk.PipelineStageFlag2}
end

SubresourceState(usage::Union{ImageUsage, AttachmentUsage}) = SubresourceState(SyncRequirements(usage), Dictionary{Vk.AccessFlag2,Vk.PipelineStageFlag2}())
SubresourceState() = SubresourceState(SyncRequirements(), Dictionary{Vk.AccessFlag2,Vk.PipelineStageFlag2}())

struct ImageResourceState
  map::SubresourceMap{SubresourceState}
end

ImageResourceState(image::Image) = ImageResourceState(SubresourceMap(image.layers, image.mip_levels, SubresourceState()))

const ResourceState = Union{BufferResourceState, ImageResourceState}

struct SynchronizationState
  resources::Dictionary{ResourceID,ResourceState}
end

SynchronizationState() = SynchronizationState(Dictionary{ResourceID,ResourceState}())

must_synchronize(sync::SyncRequirements) = !iszero(sync.stages)
must_synchronize(sync::SyncRequirements, from_layout, to_layout) = must_synchronize(sync) || from_layout ≠ to_layout

function synchronize_access!(info::Vk.DependencyInfo, state::SynchronizationState, node::RenderNode, resource::Resource, usage::BufferUsage)
  buffer_state = get!(BufferResourceState, state.resources, resource.id)::BufferResourceState
  sync = restrict_synchronization_scope(buffer_state.accesses, SyncRequirements(usage))
  must_synchronize(sync) || return
  WRITE in usage.access && (state.resources[resource.id] = BufferResourceState(usage))
  buffer = resource.data::Buffer
  barrier = Vk.BufferMemoryBarrier2(
    0,
    0,
    buffer.handle,
    buffer.offset,
    buffer.size;
    src_access_mask = buffer_state.sync.access,
    dst_access_mask = sync.access,
    src_stage_mask = buffer_state.sync.stages,
    dst_stage_mask = sync.stages,
  )
  push!(info.buffer_memory_barriers, barrier)
  log_synchronization(node, resource, usage)
end

function restrict_synchronization_scope(accesses, sync)
  remaining_sync_stages = sync.stages
  for (access, stages) in pairs(accesses)
    if covers(access, sync.access)
      remaining_sync_stages &= ~stages
    end
    iszero(remaining_sync_stages) && break
  end
  @set sync.stages = remaining_sync_stages
end

function synchronize_access!(info::Vk.DependencyInfo, state::SynchronizationState, node::RenderNode, resource::Resource, usage::Union{ImageUsage, AttachmentUsage})
  image = get_image(resource)
  image_state = get!(() -> ImageResourceState(image), state.resources, resource.id)
  subresource = Subresource(resource.data::Union{Image, ImageView, Attachment})
  subresource_state = image_state.map[subresource]
  sync = restrict_synchronization_scope(subresource_state.accesses, SyncRequirements(usage))
  to_layout = image_layout(usage.type, usage.access)
  match_subresource(image.layout, subresource) do matched_layer_range, matched_mip_range, from_layout
    must_synchronize(sync, from_layout, to_layout) || return
    image_state.map[subresource] = SubresourceState(usage)
    image.layout[subresource] = to_layout
    barrier = Vk.ImageMemoryBarrier2(
      from_layout,
      to_layout,
      0,
      0,
      image.handle,
      Vk.ImageSubresourceRange(Subresource(subresource.aspect, matched_layer_range, matched_mip_range));
      src_access_mask = subresource_state.sync.access,
      dst_access_mask = sync.access,
      src_stage_mask = subresource_state.sync.stages,
      dst_stage_mask = sync.stages,
    )
    push!(info.image_memory_barriers, barrier)
    log_synchronization(node, resource, usage)
  end
end

function log_synchronization(node::RenderNode, resource::Resource, usage)
  @debug "Synchronization: $(sprint(print_name, node)) ⇒ $(sprint(print_name, resource)) ($(usage.access == WRITE ? "write" : usage.access == READ ? "read" : usage.access))"
end

function dependency_info!(state::SynchronizationState, node_uses, resources, node::RenderNode)
  info = Vk.DependencyInfo([], [], [])
  uses = node_uses[node.id]
  for use in uses
    resource = resources[use.id]
    synchronize_access!(info, state, node, resource, use.usage)
  end
  info
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
