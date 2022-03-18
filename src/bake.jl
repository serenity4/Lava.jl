struct BakedRenderGraph
  device::Device
  global_data::GlobalData
  nodes::Vector{RenderNode}
  resources::PhysicalResources
  uses::Dictionary{NodeUUID,ResourceUses}
  image_layouts::Dictionary{UUID,Vk.ImageLayout}
end

function BakedRenderGraph(device, nodes)
  gd = GlobalData(device)
  records, pipeline_hashes = record_commands!(baked, nodes)
end

function bake(rg::RenderGraph, uses)
  gd = GlobalData(rg.device)
  uses = ResourceUses(rg)
  check_physical_resources(rg, uses)
  resources = merge(materialize_logical_resources(rg), rg.physical_resources)
  BakedRenderGraph(rg.device, gd, sort_nodes(rg), resources, uses, Dictionary())
end

function render(baked::BakedRenderGraph, command_buffer::CommandBuffer)
  create_pipelines!(device)

  # Fill command buffer with synchronization commands & recorded commands.
  initialize(command_buffer, device, baked.global_data)
  flush(command_buffer, baked, nodes, records, pipeline_hashes)
  baked
end

function record_commands!(baked::BakedRenderGraph)
  records = CompactRecord[]
  pipeline_hashes = Dictionary{ProgramInstance,UInt64}()

  # record commands and submit pipelines for creation
  for node in baked.nodes
    record = CompactRecord(baked.device, baked.global_data, node)
    node.render(record)
    push!(records, record)
    merge!(pipeline_hashes, request_pipelines(baked, record))
  end

  records, pipeline_hashes
end

function Base.flush(cb::CommandBuffer, baked::BakedRenderGraph, nodes, records, pipeline_hashes)
  binding_state = BindState()
  for (node, record) in zip(nodes, records)
    synchronize_before(cb, baked, node)
    begin_render_node(cb, baked, node)
    binding_state = flush(cb, record, baked.device, binding_state, pipeline_hashes)
    end_render_node(node, cb)
    synchronize_after(cb, baked, node)
  end
end

function rendering_info(rg::BakedRenderGraph, node::RenderNode)
  color_attachments = Vk.RenderingAttachmentInfo[]
  depth_attachment = C_NULL
  stencil_attachment = C_NULL

  for (uuid, attachment_usage) in rg.uses[node.uuid].attachments
    attachment = rg.physical_resources[uuid]
    (; aspect) = usage
    info = rendering_info(attachment, attachment_usage)
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
    node.render_area,
    0,
    0,
    color_attachments;
    depth_attachment,
    stencil_attachment,
  )
end

struct ResourceSynchronizationState
  accesses::Dictionary{Vk.AccessFlag2, Vk.PipelineStageFlag2}
  current_layout::Ref{Vk.ImageLayout}
end

ResourceSynchronizationState() = ResourceSynchronizationState(Dictionary(), Ref{Vk.ImageLayout}())

struct SynchronizationState
  node::RenderNode
  resources::Dictionary{ResourceUUID, ResourceSynchronizationState}
end

SynchronizationState(node::RenderNode) = SynchronizationState(node, Dictionary())

function synchronize(state::SynchronizationState, deps::ResourceDependency, resource::PhysicalResource)
  rstate = get!(SynchronizationState, state.resources, resource.uuid)
  
end

function dependency_info(baked::BakedRenderGraph, node::RenderNode)
  info = Vk.DependencyInfoKHR([], [], [])
  for resource in neighbors(g, pass)
    (req_access, req_stages) = access(g, pass, resource) => stages(g, pass)
    # if the resource was not written to recently, no synchronization is required
    if has_prop(g, resource, :last_write)
      (r_access, p_stages) = last_write(g, resource)
      req_access_bits = access_bits(type, req_access, req_stages)
      sync_state = synchronization_state(g, resource)
      synced_stages = Vk.PipelineStageFlag(0)
      barrier_needed = true
      for (access, stages) in pairs(sync_state)
        if covers(access, req_access_bits)
          synced_stages |= (stages & req_stages)
        end
        if synced_stages == req_stages
          barrier_needed = false
          break
        else
          # keep only stages that haven't been synced
          req_stages &= ~synced_stages
        end
      end
      if barrier_needed
        @switch class begin
          @case &RESOURCE_CLASS_BUFFER
          buff = buffer(g, resource)
          barrier = Vk.BufferMemoryBarrier2KHR(
            0, 0, handle(buff), offset(buff), size(buff);
            src_stage_mask = p_stages,
            src_access_mask = r_access,
            dst_stage_mask = req_stages,
            dst_access_mask = req_access_bits,
          )
          push!(info.buffer_memory_barriers, barrier)
          @case &RESOURCE_CLASS_IMAGE || &RESOURCE_CLASS_ATTACHMENT
          view = if class == RESOURCE_CLASS_IMAGE
            View(image(g, resource))
          else
            attachment(g, resource).view
          end
          new_layout = image_layout(g, pass, resource)
          range = subresource_range(view)
          barrier = Vk.ImageMemoryBarrier2KHR(
            current_layout(g, resource), new_layout, 0, 0, handle(view.image), range;
            src_stage_mask = p_stages,
            src_access_mask = r_access,
            dst_stage_mask = req_stages,
            dst_access_mask = req_access_bits,
          )
          push!(info.image_memory_barriers, barrier)
          set_prop!(g, resource, :current_layout, new_layout)
          view.image.layout[] = new_layout
        end
        set!(sync_state, req_access_bits, req_stages)
      end
    elseif class in (RESOURCE_CLASS_IMAGE, RESOURCE_CLASS_ATTACHMENT) && current_layout(g, resource) ≠ image_layout(g, pass, resource)
      # perform the required layout transition without further synchronization
      view = if class == RESOURCE_CLASS_IMAGE
        View(image(g, resource))
      else
        attachment(g, resource).view
      end
      new_layout = image_layout(g, pass, resource)
      barrier = Vk.ImageMemoryBarrier2KHR(current_layout(g, resource), new_layout, 0, 0, handle(view.image), subresource_range(view))
      push!(info.image_memory_barriers, barrier)
      set_prop!(g, resource, :current_layout, new_layout)
      view.image.layout[] = new_layout
    end
    if WRITE in req_access
      set_prop!(g, resource, :last_write, access_bits(resource_type(g, pass, resource), req_access, req_stages) => req_stages)
      set_prop!(g, resource, :synchronization_state, Dictionary{Vk.AccessFlag,Vk.PipelineStageFlag}())
    end
  end
  info, state
end

"""
Build barriers for all resources that require it.
"""
function synchronize_before(cb, baked::BakedRenderGraph, node::RenderNode, state::SynchronizationState)
  state, info = dependency_info(baked, node, state)
  if !isempty(info.image_memory_barriers) || !isempty(info.buffer_memory_barriers)
    Vk.cmd_pipeline_barrier_2_khr(cb, info)
  end
  state
end

function begin_render_node(cb, rg::RenderGraph, node::RenderNode)
  isnothing(node.render_pass) && return
  Vk.cmd_begin_rendering(cb, rendering_info(rg, node))
end

function end_render_node(cb, node)
  if !isnothing(node.render_pass)
    Vk.cmd_end_rendering(cb)
  end
end

function synchronize_after(cb, rg, node)
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
