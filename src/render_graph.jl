struct ResourceDependency
  type::ResourceType
  access::MemoryAccess
  clear_value::Optional{NTuple{4,Float32}}
  samples::Int64
end
ResourceDependency(type, access; clear_value = nothing, samples = 1) = ResourceDependency(type, access, clear_value, samples)

function Base.merge(x::ResourceDependency, y::ResourceDependency)
  @assert x.uuid === y.uuid
  ResourceDependency(x.uuid, x.type | y.type, x.access | y.access)
end

struct RenderArea
  rect::Vk.Rect2D
end

RenderArea(x, y) = RenderArea(Vk.Rect2D(Vk.Offset2D(0, 0), Vk.Extent2D(x, y)))
RenderArea(x, y, offset_x, offset_y) = RenderArea(Vk.Rect2D(Vk.Offset2D(offset_x, offset_y), Vk.Extent2D(x, y)))

const NodeUUID = UUID

struct RenderNode
  uuid::NodeUUID
  render
  stages::Vk.PipelineStageFlag2
  render_area::Optional{Vk.Rect2D}
end

function RenderNode(render; stages::Vk.PipelineStageFlag2 = Vk.PIPELINE_STAGE_2_ALL_COMMANDS_BIT, render_area::Optional{RenderArea} = nothing)
  RenderNode(uuid(), render, stages, isnothing(render_area) ? nothing : render_area.rect)
end

usage(::BufferAny_T, node::RenderNode, dep::ResourceDependency) =
  BufferUsage(; dep.type, dep.access, node.stages, usage = buffer_usage_bits(dep.type, dep.access))
usage(::ImageAny_T, node::RenderNode, dep::ResourceDependency) =
  ImageUsage(; dep.type, dep.access, node.stages, usage = image_usage_bits(dep.type, dep.access), dep.samples)
usage(::AttachmentAny_T, node::RenderNode, dep::ResourceDependency) = AttachmentUsage(;
  dep.type,
  dep.access,
  dep.clear_value,
  dep.samples,
  node.stages,
  usage = image_usage_bits(dep.type, dep.access),
  aspect = aspect_bits(dep.type),
)

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
  resource_graph::MetaGraph{Int64}
  nodes::Dictionary{NodeUUID,RenderNode}
  node_indices::Dictionary{NodeUUID,Int64}
  node_indices_inv::Dictionary{Int64,NodeUUID}
  resource_indices::Dictionary{ResourceUUID,Int64}
  "Resource uses per node."
  uses::Dictionary{NodeUUID,ResourceUses}
  logical_resources::LogicalResources
  physical_resources::PhysicalResources
  "Temporary resources meant to be thrown away after execution."
  temporary::Vector{UUID}
end

function RenderGraph(device::Device)
  RenderGraph(device, MetaGraph(), Dictionary(), Dictionary(), Dictionary(), Dictionary(), Dictionary(), LogicalResources(), PhysicalResources(), [])
end

device(rg::RenderGraph) = rg.device

current_layout(g, idx) = resource_attribute(g, idx, :current_layout)::Vk.ImageLayout
last_write(g, idx) = resource_attribute(g, idx, :last_write)::Pair{Vk.AccessFlag,Vk.PipelineStageFlag}
synchronization_state(g, idx) = resource_attribute(g, idx, :synchronization_state)::Dictionary{Vk.AccessFlag,Vk.PipelineStageFlag}

new_node!(rg::RenderGraph, args...; kwargs...) = add_node(rg, RenderNode(args...; kwargs...))

function add_node(rg::RenderGraph, node::RenderNode)
  (; uuid) = node
  !haskey(rg.nodes, uuid) || error("Node '$uuid' was already added to the render graph. Passes can only be provided once.")
  insert!(rg.nodes, uuid, node)
  g = rg.resource_graph
  add_vertex!(g)
  insert!(rg.node_indices, uuid, nv(g))
  insert!(rg.node_indices_inv, nv(g), uuid)
  uuid
end

new!(rg::RenderGraph, args...) = add_resource!(rg, new!(rg.logical_resources, args...))

@forward RenderGraph.logical_resources (buffer, image, attachment)

function add_resource(rg::RenderGraph, data::Union{Buffer,Image,Attachment})
  resource = new!(rg.physical_resources, data)
  push!(rg.temporary, resource.uuid)
  add_resource(rg, resource)
end

function add_resource(rg::RenderGraph, data::PhysicalResource)
  (; uuid) = data
  in(data, rg.physical_resources) && return uuid
  insert!(rg.physical_resources, uuid, data)
  add_resource(rg, uuid)
  uuid
end

function add_resource(rg::RenderGraph, data::LogicalResource)
  (; uuid) = data
  in(data, rg.logical_resources) && return uuid
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

function add_resource_dependency(rg::RenderGraph, node::RenderNode, resource, dependency::ResourceDependency)
  resource_uuid = add_resource(rg, resource)
  node_uuid = node.uuid

  # Add edge.
  haskey(rg.nodes, node_uuid) || add_node(rg, node)
  v = rg.node_indices[node_uuid]
  g = rg.resource_graph
  haskey(rg.resource_indices, resource_uuid) || add_resource(rg, resource_uuid)
  i = rg.resource_indices[resource_uuid]
  add_edge!(g, i, v)

  # Add usage.
  uses = get!(ResourceUses, rg.uses, node_uuid)
  resource_usage = usage(resource, node, dependency)
  set!(uses, resource_uuid, resource_usage)
  nothing
end

macro add_resource_dependencies(rg, ex)
  add_resource_dependencies(rg, ex)
end

function add_resource_dependencies(rg, ex::Expr)
  rg = esc(rg)
  lines = @match ex begin
    Expr(:block, _...) => ex.args
    _ => [ex]
  end

  dependency_exs = Dictionary()
  node_exs = []

  for line in lines
    line isa LineNumberNode && continue
    (f, reads, writes) = @match normalize(line) begin
      :($writes = $f($(reads...))) => (f, reads, Meta.isexpr(writes, :tuple) ? writes.args : [writes])
      _ => error("Malformed expression, expected expression of the form :((reads...) = pass(writes...)), got $line")
    end

    !in(f, node_exs) || error("Node '$f' is specified more than once")
    push!(node_exs, f)

    node_dependencies = Dictionary()

    for r in reads
      expr, type = extract_resource_spec(r)
      expr, clear_value, samples = extract_special_usage(expr)
      isnothing(clear_value) || error("Specifying a clear value for a read attachment is illegal.")
      !haskey(node_dependencies, expr) || error("Resource $expr for node $f specified multiple times in read access")
      insert!(node_dependencies, expr, (type, READ, clear_value, samples))
    end
    for w in writes
      expr, type = extract_resource_spec(w)
      expr, clear_value, samples = extract_special_usage(expr)
      access = WRITE
      if haskey(node_dependencies, expr)
        read_deps = node_dependencies[expr]
        WRITE ∉ read_deps[2] || error("Resource $expr for node $f specified multiple times in write access")
        type |= read_deps[1]
        access |= read_deps[2]
        samples = max(samples, read_deps[4])
      end
      set!(node_dependencies, expr, (type, access, clear_value, samples))
    end

    insert!(dependency_exs, f, node_dependencies)
  end

  add_dependency_exs = Expr[]
  for (node_expr, node_dependencies) in pairs(dependency_exs)
    for (expr, dependency) in pairs(node_dependencies)
      push!(add_dependency_exs, :(add_resource_dependency($rg, $(esc(node_expr)), $(esc(expr)), ResourceDependency($(dependency...)))))
    end
  end
  Expr(:block, add_dependency_exs..., rg)
end

function extract_special_usage(ex)
  clear_value = nothing
  samples = 1
  if Meta.isexpr(ex, :call) && ex.args[1] == :(=>)
    clear_value = ex.args[3]
    ex = ex.args[2]
  end
  if Meta.isexpr(ex, :call) && ex.args[1] == :(*)
    samples = ex.args[3]
    ex = ex.args[2]
  end
  ex, clear_value, samples
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

function clear_attachments(rg::RenderGraph, pass::UUID, color_clears, depth_clear = nothing, stencil_clear = nothing)
  clears = Dictionary{ResourceUUID,Vk.ClearValue}()
  for (resource, color) in color_clears
    clear_value = Vk.ClearValue(Vk.ClearColorValue(convert(NTuple{4,Float32}, color)))
    insert!(clears, resource, clear_value)
  end
  !isnothing(depth_clear)
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
  for node in rg.nodes
    dst = rg.node_indices[node.uuid]
    uses = rg.uses[node.uuid]
    for collection in (uses.buffers, uses.images, uses.attachments)
      for (uuid, usage) in pairs(collection)
        WRITE in usage.access || continue
        for src in neighbors(g, rg.resource_indices[uuid])
          src == dst && continue
          add_edge!(eg, src, dst)
        end
      end
    end
  end
  eg
end

function sort_nodes(rg::RenderGraph)
  eg = execution_graph(rg)
  !is_cyclic(eg) || error("The render graph is cyclical, cannot determine an execution order.")
  indices = topological_sort_by_dfs(eg)
  collect(rg.nodes)[indices]
end

ResourceUses(rg::RenderGraph) = merge(values(rg.uses)...)

function resolve_attachment_pairs(rg::RenderGraph)
  resolve_pairs = Dictionary{ResourceUUID, LogicalAttachment}()
  for attachment in rg.logical_resources.attachments
    if is_multisampled(attachment)
      insert!(resolve_pairs, uuid(attachment), LogicalAttachment(uuid(), attachment.format, attachment.dims))
    end
  end
  for attachment in rg.physical_resources.attachments
    if is_multisampled(attachment)
      (; info) = attachment
      insert!(resolve_pairs, uuid(info), LogicalAttachment(uuid(), info.format, info.dims))
    end
  end
  resolve_pairs
end

function add_resolve_attachments(rg::RenderGraph, resolve_pairs::Dictionary{ResourceUUID, LogicalAttachment})
  for (resource_uuid, resolve_attachment) in pairs(resolve_pairs)
    # Add resource in the render graph.
    add_resource(rg, resolve_attachment)
    i = rg.resource_indices[resource_uuid]

    # Add resource usage for all nodes used by the destination attachment.
    for j in neighbors(rg.resource_graph, i)
      attachment_uses = rg.uses[rg.node_indices_inv[j]].attachments
      usage = attachment_uses[resource_uuid]
      insert!(attachment_uses, uuid(resolve_attachment), @set usage.samples = resolve_attachment.samples)
    end
  end
end

function materialize_logical_resources(rg::RenderGraph, uses::ResourceUses)
  res = PhysicalResources()
  for info in rg.logical_resources.buffers
    usage = uses[info]
    insert!(res, info.uuid, buffer(rg.device; info.size, usage.usage))
  end
  for info in rg.logical_resources.images
    usage = uses[info]
    dims = (info.dims[1], info.dims[2])
    insert!(res, info.uuid, image(rg.device, info.format; dims, usage.usage))
  end
  for info in rg.logical_resources.attachments
    usage = uses[info]
    dims = isnothing(info.dims) ? nothing : (info.dims[1], info.dims[2])
    if isnothing(info.dims)
      # Try to inherit image dimensions from a render area in which the node is used.
      for node in rg.nodes
        !isnothing(node.render_area) || continue
        if haskey(rg.uses[node.uuid].attachments, info.uuid)
          ext = node.render_area.extent
          dims = (ext.width, ext.height)
          break
        end
      end
      isnothing(dims) && error(
        "Could not determine the dimensions of the attachment $(info.uuid). You must either provide them or use the attachment with a node that has a render area.",
      )
    end
    insert!(res, info.uuid, attachment(rg.device, info.format; dims, usage.samples, usage.aspect, usage.access, usage.usage))
  end
  res
end

function check_physical_resources(rg::RenderGraph, uses::ResourceUses)
  (; physical_resources) = rg
  for buffer in physical_resources.buffers
    usage = uses[buffer]
    usage.usage in buffer.usage || error("An existing buffer with usage $(buffer.usage) was provided, but a usage of $(usage.usage) is required.")
  end
  for image in physical_resources.images
    usage = uses[image]
    usage.usage in image.usage || error("An existing image with usage $(image.usage) was provided, but a usage of $(usage.usage) is required.")
  end
  for attachment in physical_resources.attachments
    usage = uses[attachment]
    usage.usage in attachment.usage ||
      error("An existing attachment with usage $(attachment.usage) was provided, but a usage of $(usage.usage) is required.")
    usage.samples == samples(attachment) ||
      error(
        "An existing attachment with a multisampling setting of $(samples(attachment)) samples was provided, but is used with $(usage.samples) samples.",
      )
    usage.aspect in attachment.aspect ||
      error("An existing attachment with aspect $(attachment.aspect) was provided, but is used with an aspect of $(usage.aspect).")
  end
end
