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

must_synchronize(sync_reqs::SyncRequirements) = iszero(sync_reqs.stages)
must_synchronize(sync_reqs::SyncRequirements, from_layout, to_layout) = must_synchronize(sync_reqs) || from_layout ≠ to_layout

function synchronize_buffer_access!(state::SynchronizationState, resource::Resource, usage::BufferUsage)
  rstate = get!(ResourceState, state.resources, resource.id)
  sync_reqs = restrict_sync_requirements(rstate.last_accesses, SyncRequirements(usage))
  must_synchronize(sync_reqs) || return
  WRITE in usage.access && (state.resources[resource.id] = ResourceState(usage))
  buffer = resource.data::Buffer
  Vk.BufferMemoryBarrier2(
    0,
    0,
    buffer.handle,
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
  must_synchronize(sync_reqs, from_layout, to_layout) || return
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
