struct RenderArea
  rect::Vk.Rect2D
end

RenderArea(x, y) = RenderArea(Vk.Rect2D(Vk.Offset2D(0, 0), Vk.Extent2D(x, y)))
RenderArea(x, y, offset_x, offset_y) = RenderArea(Vk.Rect2D(Vk.Offset2D(offset_x, offset_y), Vk.Extent2D(x, y)))

struct DrawInfo
  command::DrawCommand
  program::Program
  targets::RenderTargets
  state::DrawState
end

Base.@kwdef struct RenderNode
  id::NodeID = NodeID()
  stages::Vk.PipelineStageFlag2 = Vk.PIPELINE_STAGE_2_ALL_COMMANDS_BIT
  render_area::Optional{RenderArea} = nothing
  "Data required for issuing draw calls that will only be valid for a given cycle."
  draw_infos::Optional{Vector{DrawInfo}} = DrawInfo[]
  "Program invocations, that will generate [`DrawInfo`](@ref) calls at every cycle."
  program_invocations::Optional{Vector{ProgramInvocation}} = ProgramInvocation[]
end

function Base.copy(node::RenderNode)
  RenderNode(node.id, node.stages, node.render_area, isnothing(node.draw_infos) ? nothing : copy(node.draw_infos), isnothing(node.program_invocations) ? nothing : copy(node.program_invocations))
end

Descriptor(type::DescriptorType, data, node::RenderNode; flags = DescriptorFlags(0)) = Descriptor(type, data, node.id; flags)

draw(node::RenderNode, args...; kwargs...) = push!(node.draw_infos, DrawInfo(args...; kwargs...))

function ResourceUsage(resource::Resource, node::RenderNode, dep::ResourceDependency)
  usage = @match resource_type(resource) begin
    &RESOURCE_TYPE_BUFFER => BufferUsage(; dep.type, dep.access, node.stages, usage_flags = buffer_usage_flags(dep.type, dep.access))
    &RESOURCE_TYPE_IMAGE => ImageUsage(; dep.type, dep.access, node.stages, usage_flags = image_usage_flags(dep.type, dep.access), dep.samples)
    &RESOURCE_TYPE_ATTACHMENT => AttachmentUsage(;
        dep.type,
        dep.access,
        dep.clear_value,
        dep.samples,
        node.stages,
        usage_flags = image_usage_flags(dep.type, dep.access),
        aspect = aspect_flags(dep.type),
      )
  end
  ResourceUsage(resource.id, usage)
end

"""
Render graph implementation.

A render graph has a list of logical resources (buffers, images, attachments) that are
referenced by passes. They are turned into physical resources for the actual execution of those passes.

The render graph uses two graph structures: a resource graph and an execution graph. The execution graph is computed from the resource
graph during baking.

## Resource graph (bipartite, directed)

This bipartite graph has two types of vertices: passes and resources.
An edge from a resource to a pass describes a read dependency. An edge from a pass to a resource describes a write dependency.

## Execution graph (directed, acyclic)

In this graph, vertices represent passes, and edges are resource dependencies between passes.
A topological sort of this graph represents a possible sequential execution order that respects execution dependencies.

This graph is generated just-in-time and is used to convert the resource graph into a linear sequence of passes.
"""
struct RenderGraph
  device::Device
  "Used to allocate lots of tiny objects."
  allocator::LinearAllocator
  resource_graph::SimpleDiGraph{Int64}
  nodes::Dictionary{NodeID,RenderNode}
  node_indices::Dictionary{NodeID,Int64}
  node_indices_inv::Dictionary{Int64,NodeID}
  resource_indices::Dictionary{ResourceID,Int64}
  "Resource uses per node. One resource may be used several times by a single node."
  uses::Dictionary{NodeID,Dictionary{ResourceID, Vector{ResourceUsage}}}
  resources::Dictionary{ResourceID, Resource}
  "Temporary resources meant to be thrown away after execution."
  temporary::Vector{ResourceID}
end

function RenderGraph(device::Device, allocator_size = 1_000_000)
  RenderGraph(device, LinearAllocator(device, 1_000_000), SimpleDiGraph(), Dictionary(), Dictionary(), Dictionary(), Dictionary(), Dictionary(), Dictionary(), [])
end

device(rg::RenderGraph) = rg.device

function add_node!(rg::RenderGraph, node::RenderNode)
  existing = get(rg.nodes, node.id, nothing)
  if !isnothing(existing)
    existing === node || error("Trying to overwrite a node ID with a different node.")
  else
    insert!(rg.nodes, node.id, node)
  end
  g = rg.resource_graph
  add_vertex!(g)
  insert!(rg.node_indices, node.id, nv(g))
  insert!(rg.node_indices_inv, nv(g), node.id)
  nothing
end

new!(rg::RenderGraph, args...) = add_resource!(rg, new!(rg.logical_resources, args...))

@forward RenderGraph.logical_resources (buffer, image, attachment)

function add_resource!(rg::RenderGraph, resource::Resource)
  islogical(resource) && push!(rg.temporary, resource.id)

  # Add to list of resources.
  existing = get!(rg.resources, resource.id, resource)
  if !isnothing(existing)
    resource === existing || error("Trying to overwrite a resource ID with a different resource.")
  else
    insert!(rg.resources, resource.id, resource)
  end

  # Add to the resource graph.
  haskey(rg.resource_indices, resource.id) && return
  g = rg.resource_graph
  add_vertex!(g)
  insert!(rg.resource_indices, resource.id, nv(g))
  nothing
end

function add_resource_dependency!(rg::RenderGraph, node::RenderNode, resource::Resource, dependency::ResourceDependency)
  add_resource!(rg, resource)

  # Add edge.
  haskey(rg.nodes, node.id) || add_node!(rg, node)
  g = rg.resource_graph
  v = rg.resource_indices[resource.id]
  w = rg.node_indices[node.id]
  add_edge!(g, v, w)

  # Add usage.
  uses = get!(Dictionary{ResourceID,Vector{ResourceUsage}}, rg.uses, node.id)
  push!(get!(Vector{ResourceUsage}, uses, resource.id), ResourceUsage(resource, node, dependency))
  nothing
end

macro add_resource_dependencies(rg, ex)
  add_resource_dependencies!(rg, ex)
end

function add_resource_dependencies!(rg, ex::Expr)
  rg = esc(rg)
  lines = @match ex begin
    Expr(:block, _...) => ex.args
    _ => [ex]
  end
  node_exs = []
  dependency_exs = Dictionary()
  for line in lines
    isa(line, LineNumberNode) && continue
    (f, reads, writes) = @match normalize(line) begin
      :($writes = $f($(reads...))) => (f, reads, Meta.isexpr(writes, :tuple) ? writes.args : [writes])
      _ => error("Malformed expression, expected expression of the form :((reads...) = pass(writes...)), got $line")
    end
    !in(f, node_exs) || error("Node '$f' is specified more than once")
    push!(node_exs, f)
    insert!(dependency_exs, f, resource_dependencies(reads, writes))
  end

  add_dependency_exs = Expr[]
  for (node_expr, node_dependencies) in pairs(dependency_exs)
    for (expr, dependency) in pairs(node_dependencies)
      push!(add_dependency_exs, :(add_resource_dependency!($rg, $(esc(node_expr)), $(esc(expr)), ResourceDependency($(dependency...)))))
    end
  end
  Expr(:block, add_dependency_exs..., rg)
end

function resource_dependencies(reads::AbstractVector, writes::AbstractVector)
  dependencies = Dictionary()
  for r in reads
    expr, type = extract_resource_spec(r)
    expr, clear_value, samples = extract_special_usage(expr)
    isnothing(clear_value) || error("Specifying a clear value for a read attachment is not allowed.")
    !haskey(dependencies, expr) || error("Resource $expr specified multiple times in read access.")
    insert!(dependencies, expr, (type, READ, clear_value, samples))
  end
  for w in writes
    expr, type = extract_resource_spec(w)
    expr, clear_value, samples = extract_special_usage(expr)
    access = WRITE
    if haskey(dependencies, expr)
      read_deps = dependencies[expr]
      WRITE ∉ read_deps[2] || error("Resource $expr specified multiple times in write access.")
      type |= read_deps[1]
      access |= read_deps[2]
      samples = max(samples, read_deps[4])
    end
    set!(dependencies, expr, (type, access, clear_value, samples))
  end
  dependencies
end

"""

# Example

```julia
@resource_dependencies begin
  @read
  vbuffer::Buffer::Vertex
  ibuffer::Buffer::Index

  @write
  emissive => (0.0, 0.0, 0.0, 1.0)::Color
  albedo::Color
  normal::Color
  pbr::Color
  depth::Depth
end
```
"""
macro resource_dependencies(ex)
  reads = []
  writes = []
  ret = Expr(:vect)
  @match ex begin
    Expr(:block, _...) => ex.args
    _ => error("Expected block declaration, got $ex")
  end
  mode = nothing
  for line in ex.args
    isa(line, LineNumberNode) && continue
    if Meta.isexpr(line, :macrocall)
      mode = @match line.args[1] begin
        &Symbol("@read") => :read
        &Symbol("@write") => :write
        m => error("Expected call to @read or @write macro, got a call to macro $m instead")
      end
    else
      !isnothing(mode) || error("Expected @read or @write macrocall, got $line")
      mode === :read ? push!(reads, line) : mode === :write ? push!(writes, line) : error("Unexpected unknown mode '$mode'")
    end
  end
  for (expr, dependency) in pairs(resource_dependencies(reads, writes))
    push!(ret.args, :($(esc(expr)) => ResourceDependency($(dependency...))))
  end
  :(dictionary($ret))
end

function extract_special_usage(ex)
  clear_value = nothing
  samples = 1
  if Meta.isexpr(ex, :call) && ex.args[1] == :(=>)
    # Don't rely on the conversion from the constructor.
    # See https://github.com/JuliaLang/julia/issues/45485
    clear_value = :(Base.convert(NTuple{4, Float32}, $(esc(ex.args[3]))))
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
    :($r::Buffer::Vertex) => (r => SHADER_RESOURCE_TYPE_VERTEX_BUFFER)
    :($r::Buffer::Index) => (r => SHADER_RESOURCE_TYPE_INDEX_BUFFER)
    :($r::Buffer::Storage) => (r => SHADER_RESOURCE_TYPE_BUFFER | SHADER_RESOURCE_TYPE_STORAGE)
    :($r::Buffer::Uniform) => (r => SHADER_RESOURCE_TYPE_BUFFER | SHADER_RESOURCE_TYPE_UNIFORM)
    :($r::Buffer) => (r => SHADER_RESOURCE_TYPE_BUFFER)
    :($r::Color) => (r => SHADER_RESOURCE_TYPE_COLOR_ATTACHMENT)
    :($r::Depth) => (r => SHADER_RESOURCE_TYPE_DEPTH_ATTACHMENT)
    :($r::Stencil) => (r => SHADER_RESOURCE_TYPE_STENCIL_ATTACHMENT)
    :($r::Depth::Stencil) || :($_::Stencil::Depth) => (r => SHADER_RESOURCE_TYPE_DEPTH_ATTACHMENT | SHADER_RESOURCE_TYPE_STENCIL_ATTACHMENT)
    :($r::Texture) => (r => SHADER_RESOURCE_TYPE_TEXTURE)
    :($r::Image::Storage) => (r => SHADER_RESOURCE_TYPE_IMAGE | SHADER_RESOURCE_TYPE_STORAGE)
    :($r::Input) => (r => SHADER_RESOURCE_TYPE_INPUT_ATTACHMENT)
    ::Symbol => error("Resource type annotation required for $ex")
    _ => error("Invalid or unsupported resource type annotation for $ex")
  end
end

function execution_graph(rg::RenderGraph, node_uses)
  g = rg.resource_graph
  eg = SimpleDiGraph(length(rg.nodes))
  for node in rg.nodes
    dst = rg.node_indices[node.id]
    for (; id, usage) in node_uses[node.id]
      WRITE in usage.access || continue
      for src in neighbors(g, rg.resource_indices[id])
        src == dst && continue
        add_edge!(eg, src, dst)
      end
    end
  end
  eg
end

function sort_nodes(rg::RenderGraph, node_uses::Dictionary{NodeID, Dictionary{ResourceID, ResourceUsage}})
  eg = execution_graph(rg, node_uses)
  !is_cyclic(eg) || error("The render graph is cyclical, cannot determine an execution order.")
  indices = topological_sort_by_dfs(eg)
  collect(rg.nodes)[indices]
end

function add_resource_dependencies!(rg::RenderGraph, node::RenderNode)
  for invocation in node.program_invocations
    for (resource_id, resource_dependency) in pairs(invocation.resource_dependencies)
      add_resource_dependency!(rg, node, resource_id, resource_dependency)
    end
  end
end

"""
Expand all program invocations of all render nodes, generating
[`DrawInfo`](@ref) structures to be used during baking.

Render nodes will not be mutated; instead, copies which contain the
generated draw infos will be reinserted into the render graph.
"""
function generate_draw_infos!(rg::RenderGraph, node::RenderNode)
  isempty(node.program_invocations) && return
  # Do not mutate nodes so that they can be reused in other render graphs.
  generated_node = setproperties(node, (;
    draw_infos = DrawInfo[],
    program_invocations = nothing,
  ))
  for invocation in node.program_invocations
    draw_info = draw_info!(rg.allocator, rg.device.descriptors, invocation, node.id, rg.device)
    push!(generated_node.draw_infos, draw_info)
  end
  rg.nodes[node.id] = generated_node
end

function expand_program_invocations!(rg::RenderGraph)
  for node in rg.nodes
    !isnothing(node.program_invocations) || continue
    add_resource_dependencies!(rg, node)
    generate_draw_infos!(rg, node)
  end
end

function resolve_attachment_pairs(rg::RenderGraph)
  resolve_pairs = Dictionary{Resource, Resource}()
  for resource in rg.resources
    resource_type(resource) == RESOURCE_TYPE_ATTACHMENT || continue
    if islogical(resource)
      resolve_attachment = nothing
      for uses in rg.uses
        resource_uses = get(uses, resource.id, nothing)
        isnothing(resource_uses) && break
        combined_uses = reduce(merge, resource_uses)
        if combined_uses.usage.samples > 1
          attachment = resource.data::LogicalAttachment
          is_multisampled(attachment) || break
          resolve_attachment = logical_attachment(attachment.format, attachment.dims, attachment.mip_range, attachment.layer_range)
        end
      end
    else
      attachment = resource.data::Attachment
      is_multisampled(attachment) || continue
      resolve_attachment = logical_attachment(attachment.view.format, attachment.view.image.dims; attachment.view.mip_range, attachment.view.layer_range)
    end
    !isnothing(resolve_attachment) && insert!(resolve_pairs, resource, resolve_attachment)
  end
  resolve_pairs
end

function add_resolve_attachments!(rg::RenderGraph, resolve_pairs::Dictionary{Resource, Resource})
  for (resource, resolve_resource) in pairs(resolve_pairs)
    # Add resource in the render graph.
    add_resource!(rg, resolve_resource)

    # Add resource usage for all nodes used by the destination attachment.
    for j in neighbors(rg.resource_graph, rg.resource_indices[resource.id])
      uses_by_node = rg.uses[rg.node_indices_inv[j]]
      for use in uses_by_node[resource.id]
        @reset use.id = resolve_resource.id
        @reset use.usage.samples = 1
        push!(get!(Vector{ResourceUsage}, uses_by_node, resolve_resource.id), use)
      end
    end
  end
end

function materialize_logical_resources(rg::RenderGraph, combined_uses)
  res = Dictionary{ResourceID, Resource}()
  for use in combined_uses
    resource = rg.resources[use.id]
    islogical(resource) || continue
    @switch resource_type(resource) begin
      @case &RESOURCE_TYPE_BUFFER
      usage = use.usage::BufferUsage
      info = resource.data::LogicalBuffer
      insert!(res, resource.id, promote_to_physical(resource, Buffer(rg.device; info.size, usage.usage_flags)))

      @case &RESOURCE_TYPE_IMAGE
      usage = use.usage::ImageUsage
      info = resource.data::LogicalImage
      insert!(res, resource.id, promote_to_physical(resource, Image(rg.device; info.format, info.dims, usage.usage_flags, info.mip_levels, array_layers = info.layers)))

      @case &RESOURCE_TYPE_ATTACHMENT
      usage = use.usage::AttachmentUsage
      info = resource.data::LogicalAttachment
      (; dims) = info
      if isnothing(dims)
        # Try to inherit image dimensions from a render area in which the node is used.
        for node in rg.nodes
          !isnothing(node.render_area) || continue
          if haskey(rg.uses[node.id], resource.id)
            ext = node.render_area.rect.extent
            dims = [ext.width, ext.height]
            break
          end
        end
        !isnothing(dims) || error(
          "Could not determine the dimensions of the attachment $(resource.id). You must either provide them or use the attachment with a node that has a render area.",
        )
      end
      insert!(res, resource.id, promote_to_physical(resource, Attachment(rg.device; info.format, dims, usage.samples, usage.aspect, usage.access, usage.usage_flags, info.mip_range, info.layer_range)))
    end
  end
  res
end

function check_physical_resources(rg::RenderGraph, uses)
  for use in uses
    resource = rg.resources[use.id]
    isphysical(resource) || continue
    @switch resource_type(resource) begin
      @case &RESOURCE_TYPE_BUFFER
      usage = use.usage::BufferUsage
      buffer = resource.data::Buffer
      usage.usage_flags in buffer.usage_flags || error("An existing buffer with usage $(buffer.usage_flags) was provided, but a usage of $(usage.usage_flags) is required.")

      @case &RESOURCE_TYPE_IMAGE
      usage = use.usage::ImageUsage
      image = resource.data::Image
      usage.usage_flags in image.usage_flags || error("An existing image with usage $(image.usage_flags) was provided, but a usage of $(usage.usage_flags) is required.")

      @case &RESOURCE_TYPE_ATTACHMENT
      usage = use.usage::AttachmentUsage
      attachment = resource.data::Attachment
      usage.usage_flags in attachment.view.image.usage_flags ||
        error("An existing attachment with usage $(attachment.view.image.usage_flags) was provided, but a usage of $(usage.usage_flags) is required.")
      usage.samples == attachment.view.image.samples ||
        error(
          "An existing attachment with a multisampling setting of $(attachment.view.image.samples) samples was provided, but is used with $(usage.samples) samples.",
        )
      usage.aspect in attachment.view.aspect ||
        error("An existing attachment with aspect $(attachment.view.aspect) was provided, but is used with an aspect of $(usage.aspect).")
    end
  end
end
