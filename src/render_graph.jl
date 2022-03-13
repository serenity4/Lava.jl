struct ResourceDependency
  uuid::ResourceUUID
  type::ResourceType
  access::MemoryAccess
end

function Base.merge(x::ResourceDependency, y::ResourceDependency)
  @assert x.uuid === y.uuid
  ResourceDependency(x.uuid, x.type | y.type, x.access | y.access)
end

struct RenderArea
  rect::Vk.Rect2D
end

RenderArea(x, y) = RenderArea(Vk.Offset2D(0, 0), Vk.Extent2D(x, y))
RenderArea(x, y, offset_x, offset_y) = RenderArea(Vk.Offset2D(offset_x, offset_y), Vk.Extent2D(x, y))

const NodeUUID = UUID

struct RenderNode
  uuid::NodeUUID
  render
  stages::Vk.PipelineStageFlag
  render_area::Optional{Vk.Rect2D}
end

function RenderNode(render; stages::Vk.PipelineStageFlag2 = PIPELINE_STAGE_2_ALL_COMMANDS_BIT, render_area::Optional{RenderArea} = nothing)
  RenderNode(uuid(), render, stages, render_area.rect)
end

struct ResourceDependencies
  node::Dictionary{NodeUUID,Dictionary{ResourceUUID,ResourceDependency}}
end

@forward ResourceDependencies.node (Base.getindex, Base.delete!, Base.get, Base.haskey)

Base.merge!(x::ResourceDependencies, y::ResourceDependencies) = mergewith!(merge_node_dependencies, x.node, y.node)
merge_node_dependencies!(x, y) = mergewith!(merge, x, y)

ResourceDependencies() = ResourceDependencies(Dictionary())

"""
Frame graph implementation.

A frame graph has a list of virtual resources (buffers, images, attachments) that are
referenced by passes. They are turned into physical resources for the actual execution of those passes.

The frame graph uses two graph structures: a resource graph and an execution graph.

## Resource graph (bipartite, directed)

This bipartite graph has two types of vertices: passes and resources.
An edge from a resource to a pass describes a read dependency. An edge from a pass to a resource describes a write dependency.

Graph attributes:
- Resources:
    - `:name`: name of the resource
    - `:format` (if the resource describes either an image or an attachment)
    - `:usage`
    - `:size` (if the resource describes a buffer)
    - `:vresource`: description as a virtual resource
    - `:presource`: physical resource
    - `:current_layout` (if the resource describes an image)
    - `:last_write`: access bits and pipeline stages associated to the latest pass that wrote to the resource
    - `:synchronization_state`: access bits and pipeline stages of all passes that required to synchronize with this resource
        since the last write.
- Passes:
    - `:name`: name of the pass
- Edge between a resource and a pass (all directions)
    - `:image_layout` (if the resource describes an image)
    - `:usage`
    - `:aspect`
    - `:stage`: stages in which the resource is used
    - `:clear_value`: clear value, if applicable. If this property is absent and a load operation takes place on this resource in this pass,
        then the contents are preserved.

## Execution graph (directed, acyclic)

In this graph, vertices represent passes, and edges are resource dependencies between passes.
A topological sort of this graph represents a possible sequential execution order that respects execution dependencies.

Reusing the example above, the graph has three vertices: `gbuffer`, `lighting` and `adapt_luminance`.
`gbuffer` has five outgoing edges to `lighting`, each edge being labeled with a resource.
`lighting` has one outgoing edge to `adapt_luminance`.

This graph is generated just-in-time, to convert the resource graph into a linear sequence of passes.
"""
struct RenderGraph
  device::Device
  resource_graph::MetaGraph{Int}
  nodes::Dictionary{NodeUUID,RenderNode}
  node_indices::Dictionary{NodeUUID,Int}
  node_indices_inv::Dictionary{Int,NodeUUID}
  resource_indices::Dictionary{ResourceUUID,Int}
  "Combined use of render graph resources."
  uses::Dictionary{NodeUUID,ResourceUses}
  "Per-node resource dependencies."
  resource_dependencies::ResourceDependencies
  logical_resources::LogicalResources
  physical_resources::PhysicalResources
  "Temporary resources meant to be thrown away after execution."
  temporary::Vector{UUID}
  "Execution dependencies that must be preserved during execution."
  dependencies::Vector{Any}
end

function RenderGraph(device::Device)
  RenderGraph(device, MetaGraph(), Dictionary(), Dictionary(), Dictionary(), Dictionary())
end

device(rg::RenderGraph) = rg.device

current_layout(g, idx) = resource_attribute(g, idx, :current_layout)::Vk.ImageLayout
last_write(g, idx) = resource_attribute(g, idx, :last_write)::Pair{Vk.AccessFlag,Vk.PipelineStageFlag}
synchronization_state(g, idx) = resource_attribute(g, idx, :synchronization_state)::Dictionary{Vk.AccessFlag,Vk.PipelineStageFlag}

new_node!(rg::RenderGraph, args...; kwargs...) = add_node!(rg, RenderNode(args...; kwargs...))

function add_node!(rg::RenderGraph, node::RenderNode)
  (; uuid) = node
  !haskey(rg.nodes, uuid) || error("Node '$uuid' was already added to the frame graph. Passes can only be provided once.")
  insert!(rg.nodes, uuid, node)
  g = rg.resource_graph
  add_vertex!(g)
  insert!(rg.node_indices, uuid, nv(g))
  insert!(rg.node_indices_inv, nv(g), uuid)
  uuid
end

new_resource!(rg::RenderGraph, args...) = add_resource!(rg, new!(rg.logical_resources, args...))
new!(rg::RenderGraph, args...) = new_resource!(rg, args...)

function add_resource(rg::RenderGraph, data::Union{Buffer,Image,Attachment})
  resource = new!(rg.physical_resources, data)
  push!(rg.temporary, resource.uuid)
  add_resource(rg, resource)
end

function add_resource(rg::RenderGraph, data::PhysicalResource)
  (; uuid) = data
  haskey(rg.physical_resources, uuid) && return uuid
  insert!(rg.physical_resources, uuid, data)
  add_resource(rg, uuid)
  uuid
end

function add_resource(rg::RenderGraph, data::LogicalResource)
  (; uuid) = data
  haskey(rg.logical_resources, uuid) && return uuid
  insert!(rg.logical_resources, uuid, data)
  push!(rg.temporary, uuid)
  add_resource(rg, uuid)
  uuid
end

function add_resource(rg::RenderGraph, uuid::ResourceUUID)
  haskey(rg.resource_indices, uuid) && return uuid
  g = rg.resource_graph
  add_vertex!(g)
  insert!(rg.resource_indices, uuid, nv(g))
  uuid
end

function add_resource_dependencies(rg::RenderGraph, resource_dependencies::ResourceDependencies)
  for (node_uuid, dependencies) in pairs(resource_dependencies)
    add_resource_dependencies!(rg, rg.nodes[node_uuid], dependencies)
  end
end

function add_resource_dependencies(rg::RenderGraph, node::RenderNode, uses::Dictionary{ResourceUUID,ResourceDependency})
  for dependency in uses
    add_resource_dependency!(rg, node, dependency)
  end
end

function add_resource_dependency(rg::RenderGraph, node::RenderNode, dependency::ResourceDependency)
  node_uuid = node.uuid
  haskey(rg.nodes, node_uuid) || add_node!(rg, node)
  v = rg.node_indices[node]
  g = rg.resource_graph
  resource_uuid = dependency.uuid
  haskey(rg.resource_indices, resource_uuid) || add_resource!(rg, resource_uuid)
  i = rg.resource_indices[resource_uuid]
  add_edge!(g, i, v)
  prev_dependency = get(rg.resource_dependencies[node_uuid], resource_uuid, ResourceDependency(resource_uuid, ResourceType(0), MemoryAccess(0)))
  set!(rg.resource_dependencies[node_uuid], resource_uuid, dependency | prev_dependency)
  nothing
end

macro add_resource_dependencies(rg, ex)
  add_resource_dependencies(rg, ex)
end

macro add_resource_dependencies(rg, resource_scope, pass_scope, ex)
  add_resource_dependencies(rg, ex, resource_scope, pass_scope)
end

function add_resource_dependencies(rg, ex::Expr, resource_scope = nothing, node_scope = nothing)
  rg = esc(rg)
  lines = @match ex begin
    Expr(:block, _...) => ex.args
    _ => [ex]
  end

  dependency_exs = Dictionary{Union{Expr,Symbol},Dictionary{Union{Expr,Symbol},Pair{ResourceType,MemoryAccess}}}()
  node_exs = Set(Union{Expr,Symbol}[])
  resource_exs = Set(Union{Expr,Symbol}[])

  for line in lines
    line isa LineNumberNode && continue
    (f, reads, writes) = @match line begin
      :($writes = $f($(reads...))) => (f, reads, Meta.isexpr(writes, :tuple) ? writes.args : [writes])
      _ => error("Malformed expression, expected :(a, b = f(c, d)), got $line")
    end

    if !isnothing(resource_scope)
      for (i, w) in enumerate(writes)
        w isa Symbol && (writes[i] = :($resource_scope.$w))
      end
      for (i, r) in enumerate(reads)
        r isa Symbol && (reads[i] = :($resource_scope.$r))
      end
    end
    !isnothing(node_scope) && isa(f, Symbol) && (f = :($node_scope.$f))
    !in(f, node_exs) || error("Node '$f' is specified more than once")
    push!(node_exs, f)

    node_dependencies = Dictionary{Any,Pair{ResourceType,MemoryAccess}}()

    for (expr, type) in extract_resource_spec.(reads)
      !haskey(node_dependencies, expr) || error("Resource $expr for node $f specified multiple times in read access")
      insert!(node_dependencies, expr, type => READ)
      push!(resource_exs, expr)
    end
    for (expr, type) in extract_resource_spec.(writes)
      dep = if haskey(node_dependencies, expr)
        WRITE ∉ node_dependencies[expr].second || error("Resource $expr for node $f specified multiple times in write access")
        (type | node_dependencies[expr].first) => (WRITE | READ)
      else
        type => WRITE
      end
      set!(node_dependencies, expr, dep)
      push!(resource_exs, expr)
    end

    insert!(dependency_exs, f, node_dependencies)
  end

  declarations_map = Dictionary{Any,Symbol}()
  resource_declarations = Expr(:block)
  for expr in resource_exs
    uuid_var = gensym(:resource)
    push!(resource_declarations, :($uuid_var = add_resource($rg, $(esc(expr)))))
    insert!(declarations_map, expr, uuid_var)
  end
  add_dependency_exs = Expr[]
  for (node_expr, node_dependencies) in pairs(dependency_exs)
    for (expr, dependency) in pairs(node_dependencies)
      uuid_var = declarations_map[expr]
      push!(add_dependency_exs, :(add_resource_dependency($rg, $(esc(node_expr)), ResourceDependency($uuid_var, $(dependency...)))))
    end
  end
  Expr(:block, resource_declarations..., add_dependency_exs...)
end

function extract_resource_spec(ex::Expr)
  @match ex begin
    :($r::Buffer::Vertex) => (r => RESOURCE_TYPE_VERTEX_BUFFER)
    :($r::Buffer::Index) => (r => RESOURCE_TYPE_INDEX_BUFFER)
    :($r::Buffer::Storage) => (r => RESOURCE_TYPE_BUFFER | RESOURCE_TYPE_STORAGE)
    :($r::Buffer::Uniform) => (r => RESOURCE_TYPE_BUFFER | RESOURCE_TYPE_UNIFORM)
    :($r::Buffer) => (r => RESOURCE_TYPE_BUFFER)
    :($r::Color) => (r => RESOURCE_TYPE_COLOR_ATTACHMENT)
    :($r::Depth) => (r => RESOURCE_TYPE_DEPTH_ATTACHMENT)
    :($r::Stencil) => (r => RESOURCE_TYPE_STENCIL_ATTACHMENT)
    :($r::Depth::Stencil) || :($_::Stencil::Depth) => (r => RESOURCE_TYPE_DEPTH_ATTACHMENT | RESOURCE_TYPE_STENCIL_ATTACHMENT)
    :($r::Texture) => (r => RESOURCE_TYPE_TEXTURE)
    :($r::Image::Storage) => (r => RESOURCE_TYPE_IMAGE | RESOURCE_TYPE_STORAGE)
    :($r::Input) => (r => RESOURCE_TYPE_INPUT_ATTACHMENT)
    ::Symbol => error("Resource type annotation required for $ex")
    _ => error("Invalid or unsupported resource type annotation for $ex")
  end
end

function clear_attachments(rg::RenderGraph, pass::UUID, color_clears, depth_clears = [], stencil_clears = [])
  clears = Dictionary{Symbol,Vk.ClearValue}()
  for (resource, color) in color_clears
    clear_value = Vk.ClearValue(Vk.ClearColorValue(convert(NTuple{4,Float32}, color)))
    insert!(clears, resource, clear_value)
  end
  for (resource, depth) in depth_clears
    clear_value = Vk.ClearValue(Vk.ClearDepthStencilValue(depth, 0))
    insert!(clears, resource, clear_value)
  end
  for (resource, stencil) in stencil_clears
    clear_value = Vk.ClearValue(Vk.ClearDepthStencilValue(0.0, stencil))
    insert!(clears, resource, clear_value)
  end
  for (resource, clear_value) in pairs(clears)
    set_attribute(rg, pass, resource, :clear_value, clear_value)
  end
end

function execution_graph(rg::RenderGraph)
  g = rg.resource_graph
  eg = SimpleDiGraph(length(rg.nodes))
  for pass in rg.node_indices
    resources = neighbors(g, pass)
    for resource in resources
      WRITE in access(g, pass, resource) || continue
      for dependent_pass in neighbors(g, resource)
        dependent_pass == pass && continue
        add_edge!(eg, dependent_pass, pass)
      end
    end
  end
  eg
end

"""
Submit rendering commands to a device.

A command buffer is recorded, which may be split into multiple ones to take advantage of multithreading,
and is then submitted them to the provided device. A custom primary command buffer can be optionally passed as a keyword,
mostly intended for debugging purposes.

A semaphore to wait for can be provided to synchronize with other commands.
"""
function render(rg::RenderGraph; semaphore = nothing, command_buffer = request_command_buffer(rg.device, Vk.QUEUE_GRAPHICS_BIT), submit = true)
  analyze!(rg)
  baked = bake(rg, command_buffer)

  # submit rendering work
  submit || return command_buffer
  wait_semaphores = device.transfer_ops
  !isnothing(semaphore) && push!(wait_semaphores, semaphore)
  submit_info = Vk.SubmitInfo2KHR(wait_semaphores, [Vk.CommandBufferSubmitInfoKHR(command_buffer)], [])
  Lava.submit(device, command_buffer.queue_family_index, [submit_info]; signal_fence = true, release_after_completion = [Ref(rg)])
end

function analyze!(rg::RenderGraph)
  resolve_attributes!(rg)
  create_physical_resources!(rg)
end

function sort_nodes(rg::RenderGraph)
  indices = topological_sort_by_dfs(execution_graph(rg))
  node_uuids = collect(keys(rg.node_indices))[indices]
  map(node_uuids) do uuid
    rg.nodes[uuid]
  end
end

"""
Build barriers for all resources that require it.

Requires the extension `VK_KHR_synchronization2`.
"""
function synchronize_before(cb, rg::RenderGraph, pass::Integer)
  g = rg.resource_graph
  deps = Vk.DependencyInfoKHR([], [], [])
  for resource in neighbors(g, pass)
    type = resource_type(g, pass, resource)
    class = resource_class(g, resource)
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
          push!(deps.buffer_memory_barriers, barrier)
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
          push!(deps.image_memory_barriers, barrier)
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
      push!(deps.image_memory_barriers, barrier)
      set_prop!(g, resource, :current_layout, new_layout)
      view.image.layout[] = new_layout
    end
    if WRITE in req_access
      set_prop!(g, resource, :last_write, access_bits(resource_type(g, pass, resource), req_access, req_stages) => req_stages)
      set_prop!(g, resource, :synchronization_state, Dictionary{Vk.AccessFlag,Vk.PipelineStageFlag}())
    end
  end
  (!isempty(deps.memory_barriers) || !isempty(deps.image_memory_barriers) || !isempty(deps.buffer_memory_barriers)) &&
    Vk.cmd_pipeline_barrier_2_khr(cb, deps)
end

function rendering_info(rg::RenderGraph, node::RenderNode)
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

function begin_render_node(cb, rg::RenderGraph, node::RenderNode)
  isnothing(node.render_pass) && return
  Vk.cmd_begin_rendering(cb, rendering_info(rg, node))
end

function synchronize_after(cb, rg, node)
  nothing
end

"""
Deduce the Vulkan usage, layout and access flags form a resource given its type, stage and access.

The idea is to reconstruct information like `Vk.ACCESS_COLOR_ATTACHMENT_READ_BIT` and `Vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` from a more decoupled description.
"""
function image_layout(type::ResourceType, access::MemoryAccess, stage::Vk.PipelineStageFlag)
  @match (type, access) begin
    (&RESOURCE_TYPE_COLOR_ATTACHMENT, &READ) => Vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    (&RESOURCE_TYPE_COLOR_ATTACHMENT, &WRITE) => Vk.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    (&RESOURCE_TYPE_COLOR_ATTACHMENT || &RESOURCE_TYPE_IMAGE || RESOURCE_TYPE_TEXTURE, &(READ | WRITE)) => Vk.IMAGE_LAYOUT_GENERAL
    (&RESOURCE_TYPE_DEPTH_ATTACHMENT, &READ) => Vk.IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL
    (&RESOURCE_TYPE_DEPTH_ATTACHMENT, &WRITE) => Vk.IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
    (&RESOURCE_TYPE_STENCIL_ATTACHMENT, &READ) => Vk.IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL
    (&RESOURCE_TYPE_STENCIL_ATTACHMENT, &WRITE) => Vk.IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL
    (&(RESOURCE_TYPE_DEPTH_ATTACHMENT | RESOURCE_TYPE_STENCIL_ATTACHMENT), &READ) => Vk.IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
    (&(RESOURCE_TYPE_DEPTH_ATTACHMENT | RESOURCE_TYPE_STENCIL_ATTACHMENT), &WRITE) => Vk.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    (&RESOURCE_TYPE_INPUT_ATTACHMENT || &RESOURCE_TYPE_TEXTURE || &RESOURCE_TYPE_IMAGE, &READ) => Vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    _ => error("Unsupported combination of type $type and access $access")
  end
end

function buffer_usage_bits(type::ResourceType, access::MemoryAccess)
  bits = Vk.BufferUsageFlag(0)

  RESOURCE_TYPE_BUFFER | RESOURCE_TYPE_STORAGE in type && (bits |= Vk.BUFFER_USAGE_STORAGE_BUFFER_BIT)
  RESOURCE_TYPE_VERTEX_BUFFER in type && (bits |= Vk.BUFFER_USAGE_VERTEX_BUFFER_BIT)
  RESOURCE_TYPE_INDEX_BUFFER in type && (bits |= Vk.BUFFER_USAGE_INDEX_BUFFER_BIT)
  RESOURCE_TYPE_BUFFER in type && access == READ && (bits |= Vk.BUFFER_USAGE_UNIFORM_BUFFER_BIT)

  bits
end

function image_usage_bits(type::ResourceType, access::MemoryAccess)
  bits = Vk.ImageUsageFlag(0)

  RESOURCE_TYPE_COLOR_ATTACHMENT in type && (bits |= Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
  (RESOURCE_TYPE_DEPTH_ATTACHMENT in type || RESOURCE_TYPE_STENCIL_ATTACHMENT in type) && (bits |= Vk.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
  RESOURCE_TYPE_INPUT_ATTACHMENT in type && (bits |= Vk.IMAGE_USAGE_INPUT_ATTACHMENT_BIT)
  RESOURCE_TYPE_TEXTURE in type && (bits |= Vk.IMAGE_USAGE_SAMPLED_BIT)
  RESOURCE_TYPE_IMAGE in type && WRITE in access && (bits |= Vk.IMAGE_USAGE_STORAGE_BIT)

  bits
end

const SHADER_STAGES = |(
  Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT_KHR,
  Vk.PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT_KHR,
  Vk.PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT_KHR,
  Vk.PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT_KHR,
  Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT_KHR,
  Vk.PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR,
)

function access_bits(type::ResourceType, access::MemoryAccess, stage::Vk.PipelineStageFlag2)
  bits = Vk.AccessFlag(0)
  RESOURCE_TYPE_VERTEX_BUFFER in type && (bits |= Vk.ACCESS_VERTEX_ATTRIBUTE_READ_BIT)
  RESOURCE_TYPE_INDEX_BUFFER in type && (bits |= Vk.ACCESS_INDEX_READ_BIT)
  if RESOURCE_TYPE_COLOR_ATTACHMENT in type
    READ in access && (bits |= Vk.ACCESS_COLOR_ATTACHMENT_READ_BIT)
    WRITE in access && (bits |= Vk.ACCESS_COLOR_ATTACHMENT_WRITE_BIT)
  end
  if (RESOURCE_TYPE_DEPTH_ATTACHMENT in type || RESOURCE_TYPE_STENCIL_ATTACHMENT in type)
    #TODO: support mixed access modes (depth write, stencil read and vice-versa)
    READ in access && (bits |= Vk.ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT)
    WRITE in access && (bits |= Vk.ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
  end
  RESOURCE_TYPE_INPUT_ATTACHMENT in type && (bits |= Vk.ACCESS_INPUT_ATTACHMENT_READ_BIT)
  if RESOURCE_TYPE_BUFFER in type && !iszero(stage & SHADER_STAGES)
    access == READ && (bits |= Vk.ACCESS_UNIFORM_READ_BIT)
    WRITE in access && (bits |= Vk.ACCESS_SHADER_WRITE_BIT)
  end
  RESOURCE_TYPE_TEXTURE in type && READ in access && (bits |= Vk.ACCESS_SHADER_READ_BIT)
  RESOURCE_TYPE_TEXTURE in type && WRITE in access && (bits |= Vk.ACCESS_SHADER_WRITE_BIT)
  bits
end

function aspect_bits(type::ResourceType)
  bits = Vk.ImageAspectFlag(0)
  RESOURCE_TYPE_COLOR_ATTACHMENT in type && (bits |= Vk.IMAGE_ASPECT_COLOR_BIT)
  RESOURCE_TYPE_DEPTH_ATTACHMENT in type && (bits |= Vk.IMAGE_ASPECT_DEPTH_BIT)
  RESOURCE_TYPE_STENCIL_ATTACHMENT in type && (bits |= Vk.IMAGE_ASPECT_STENCIL_BIT)
  bits
end

function descriptor_type(type::ResourceType, access::MemoryAccess)
  @match access, type begin
    &(RESOURCE_TYPE_IMAGE | RESOURCE_TYPE_STORAGE) => Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE
    &(RESOURCE_TYPE_TEXEL | RESOURCE_TYPE_BUFFER) => Vk.DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER
    &(RESOURCE_TYPE_UNIFORM | RESOURCE_TYPE_BUFFER) => Vk.DESCRIPTOR_TYPE_UNIFORM_BUFFER
    &(RESOURCE_TYPE_UNIFORM | RESOURCE_TYPE_BUFFER | RESOURCE_TYPE_DYNAMIC) => Vk.DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
    &(RESOURCE_TYPE_UNIFORM | RESOURCE_TYPE_TEXEL | RESOURCE_TYPE_BUFFER) => Vk.DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
    &(RESOURCE_TYPE_STORAGE | RESOURCE_TYPE_BUFFER) => Vk.DESCRIPTOR_TYPE_STORAGE_BUFFER
    &(RESOURCE_TYPE_STORAGE | RESOURCE_TYPE_BUFFER | RESOURCE_TYPE_DYNAMIC) => Vk.DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC
    &RESOURCE_TYPE_TEXTURE => Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE
    &(RESOURCE_TYPE_TEXTURE | ESOURCE_TYPE_SAMPLER) => Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
    &RESOURCE_TYPE_SAMPLER => Vk.DESCRIPTOR_TYPE_SAMPLER
    _ => error("Unsupported combination of type $type and access $access")
  end
end

function ResourceUses(rg::RenderGraph)
  uses = ResourceUses()
  for (resource_uuid, i) in pairs(rg.resource_indices)
    resource = rg.logical_resources[resource_uuid]
    usage = nothing
    for j in neighbors(rg.resource_graph, i)
      node_uuid = rg.node_indices_inv[j]
      node = rg.nodes[node_uuid]
      node_usage = rg.uses[node_uuid][resource]
      if isnothing(usage)
        usage = node_usage
      elseif usage isa BufferUsage
        usage = setproperties(
          usage,
          (;
            type = usage.type | node_usage.type,
            access = usage.access | node_usage.access,
            stages = usage.stages | node.stages,
          ),
        )
      elseif usage isa ImageUsage
        usage = setproperties(
          usage,
          (;
            type = usage.type | node_usage.type,
            access = usage.access | node_usage.access,
            stages = usage.stages | node.stages,
            layout = usage.layout,
          ),
        )
      elseif usage isa AttachmentUsage
        usage = setproperties(
          usage,
          (;
            type = usage.type | node_usage.type,
            access = usage.access | node_usage.access,
            stages = usage.stages | node.stages,
            aspect = usage.aspect | aspect_bits(node_usage.type),
            samples = usage.samples | Vk.SampleCountFlag(node_usage.samples),
          ),
        )
      end
    end
    usage isa ImageUsage && (usage = (@set usage.usage = image_usage_bits(usage.type, usage.access)))
    usage isa BufferUsage && (usage = (@set usage.usage = buffer_usage_bits(usage.type, usage.access)))
    insert!(uses, resource_uuid, usage::Union{BufferUsage,ImageUsage,AttachmentUsage})
  end
  uses
end

function materialize_physical_resources!(rg::RenderGraph)
  (; logical_resources, physical_resources) = rg
  uses = ResourceUses(rg)
  check_physical_resources(rg, uses)
  for info in logical_resources.buffers
    usage = uses.buffers[info.uuid]
    insert!(physical_resources.buffers, uuid, buffer(rg.device; info.size, usage.usage))
  end
  for info in logical_resources.images
    usage = uses.images[info.uuid]
    insert!(physical_resources.images, uuid, image(rg.device; info.dims, usage.usage))
  end
  for info in logical_resources.attachments
    usage = uses.attachments[info.uuid]
    insert!(physical_resources.attachments, uuid, attachment(rg.device; info.format, usage.samples, usage.aspect, usage.access))
  end
  empty!(rg.logical_resources)
end

function check_physical_resources(rg::RenderGraph, uses::ResourceUses)
  (; physical_resources) = rg
  for buffer in physical_resources.buffers
    usage = uses.buffers[buffer.uuid]
    usage.usage in buffer.usage || error("An existing buffer with usage $(buffer.usage) was provided, but a usage of $(usage.usage) is required.")
  end
  for image in physical_resources.images
    usage = uses.images[image.uuid]
    usage.usage in image.usage || error("An existing image with usage $(image.usage) was provided, but a usage of $(usage.usage) is required.")
  end
  for attachment in physical_resources.attachments
    usage = uses.attachments[attachment.uuid]
    usage.usage in attachment.usage ||
      error("An existing attachment with usage $(attachment.usage) was provided, but a usage of $(usage.usage) is required.")
    usage.samples in attachment.samples ||
      error(
        "An existing attachment compatible with multisampling settings $(attachment.samples) samples was provided, but is used with a multisampling of $(usage.samples).",
      )
    usage.aspect in attachment.aspect ||
      error("An existing attachment with aspect $(attachment.aspect) was provided, but is used with an aspect of $(usage.aspect).")
  end
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

function ResourceClass(type::ResourceType)
  RESOURCE_TYPE_TEXTURE in type && return RESOURCE_CLASS_IMAGE
  RESOURCE_TYPE_IMAGE in type && return RESOURCE_CLASS_IMAGE
  RESOURCE_TYPE_INPUT_ATTACHMENT in type && return RESOURCE_CLASS_ATTACHMENT
  RESOURCE_TYPE_COLOR_ATTACHMENT in type && return RESOURCE_CLASS_ATTACHMENT
  RESOURCE_TYPE_DEPTH_ATTACHMENT in type && return RESOURCE_CLASS_ATTACHMENT
  RESOURCE_TYPE_STENCIL_ATTACHMENT in type && return RESOURCE_CLASS_ATTACHMENT
  RESOURCE_TYPE_BUFFER in type && return RESOURCE_CLASS_BUFFER
  RESOURCE_TYPE_VERTEX_BUFFER in type && return RESOURCE_CLASS_BUFFER
  RESOURCE_TYPE_INDEX_BUFFER in type && return RESOURCE_CLASS_BUFFER
  error("Resource type '$type' does not belong to any resource class.")
end
