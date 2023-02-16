contains_fragment_stage(stages::Vk.PipelineStageFlag2) = in(Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, stages) || in(Vk.PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, stages) || in(Vk.PIPELINE_STAGE_2_ALL_COMMANDS_BIT, stages)

Base.@kwdef struct RenderNode
  id::NodeID = NodeID()
  stages::Vk.PipelineStageFlag2 = Vk.PIPELINE_STAGE_2_ALL_COMMANDS_BIT
  render_area::Optional{RenderArea} = nothing
  commands::Optional{Vector{Command}} = Command[]
  function RenderNode(id, stages, render_area, commands)
    !iszero(stages) || throw(ArgumentError("At least one pipeline stage must be provided."))
    !isnothing(render_area) && !contains_fragment_stage(stages) && throw(ArgumentError("The fragment shader stage must be set when a `RenderArea` is provided."))
    isnothing(render_area) && contains_fragment_stage(stages) && throw(ArgumentError("The render area must be set when the fragment shader stage is included."))
    new(id, stages, render_area, commands)
  end
end

Descriptor(type::DescriptorType, data, node::RenderNode; flags = DescriptorFlags(0)) = Descriptor(type, data, node.id; flags)

function ResourceUsage(resource::Resource, node::RenderNode, dep::ResourceDependency)
  if dep.type in (RESOURCE_USAGE_COLOR_ATTACHMENT, RESOURCE_USAGE_DEPTH_ATTACHMENT, RESOURCE_USAGE_STENCIL_ATTACHMENT)
    contains_fragment_stage(node.stages) || throw(ArgumentError("Color, depth and stencil attachments are only allowed in render nodes which execute fragment shaders; for other uses, such as within compute shaders, use a generic texture or image type instead."))
  end
  usage = @match resource_type(resource) begin
    &RESOURCE_TYPE_BUFFER => BufferUsage(; dep.type, dep.access, stages = stage_flags(node, resource), usage_flags = buffer_usage_flags(dep.type, dep.access))
    &RESOURCE_TYPE_IMAGE => ImageUsage(; dep.type, dep.access, stages = stage_flags(node, resource), usage_flags = image_usage_flags(dep.type, dep.access), dep.samples)
    &RESOURCE_TYPE_ATTACHMENT => AttachmentUsage(;
        dep.type,
        dep.access,
        dep.clear_value,
        dep.samples,
        stages = stage_flags(node, resource),
        usage_flags = image_usage_flags(dep.type, dep.access),
        aspect = aspect_flags(resource.data::Union{LogicalAttachment,Attachment}),
      )
  end
  ResourceUsage(resource.id, dep.type, usage)
end

function stage_flags(node::RenderNode, resource::Resource)
  foldl(|, stage_flags(command, resource, node) for command in node.commands; init = Vk.PIPELINE_STAGE_2_NONE)
end

function stage_flags(command::Command, resource::Resource)
  command.type == COMMAND_TYPE_DISPATCH_INDIRECT && (command.compute.dispatch::DispatchIndirect).buffer.id == resource.id && return Vk.PIPELINE_STAGE_2_DRAW_INDIRECT_BIT
  command.type == COMMAND_TYPE_DRAW_INDIRECT && (command.graphics.draw::DrawIndirect).parameters.id == resource.id && return Vk.PIPELINE_STAGE_2_DRAW_INDIRECT_BIT
  if command.type == COMMAND_TYPE_TRANSFER
    (; transfer) = command
    if resource.id in (transfer.src, transfer.dst)
      is_copy(command) && return Vk.PIPELINE_STAGE_2_COPY_BIT
      is_resolve(command) && return Vk.PIPELINE_STAGE_2_RESOLVE_BIT
      is_blit(command) && return Vk.PIPELINE_STAGE_2_BLIT_BIT
      @assert false
    end
  end
  Vk.PIPELINE_STAGE_2_NONE
end

function stage_flags(targets::RenderTargets, render_state::RenderState, resource::Resource)
  stages = Vk.PIPELINE_STAGE_2_NONE
  (; color, depth, stencil) = targets
  any(att.id == resource.id for att in color) && (stages |= Vk.PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)
  depth_stencil_stages = Vk.PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | Vk.PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
  !isnothing(depth) && resource.id == depth.id && render_state.enable_depth_testing && (stages |= depth_stencil_stages)
  !isnothing(stencil) && resource.id == stencil.id && render_state.enable_stencil_testing && (stages |= depth_stencil_stages)
  stages
end

function stage_flags(command::Command, resource::Resource, node::RenderNode)
  stages = stage_flags(command, resource)
  if is_graphics(command) && isattachment(resource)
    (; graphics) = command
    stages |= stage_flags(graphics.targets, graphics.state.render_state, resource)
  end
  ifelse(iszero(stages), node.stages, stages)
end

"""
Render graph implementation.

A render graph has a list of logical resources (buffers, images, attachments) that are
referenced by passes. They are turned into physical resources for the actual execution of those passes.

The render graph uses two graph structures: a resource graph and an execution graph. The execution graph is computed from the resource
graph during baking.

!!! note
    The execution order will be computed from read and write dependencies. Any node reading a given resource will be executed after all nodes which
    have written to it. This disallows the *reuse* of resources; if you want to write to a resource, then read it, and finally write to it again, *you must use
    a new resource* for the latter. It is planned that resource optimizations will be performed exclusively by the render graph, and should automatically reuse
    the resource that was written to in this specific situation.

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

function RenderGraph(device::Device; linear_allocator_size = 2^20 #= 1_000_000 KiB =#)
  RenderGraph(device, LinearAllocator(device, linear_allocator_size), SimpleDiGraph(), Dictionary(), Dictionary(), Dictionary(), Dictionary(), Dictionary(), Dictionary(), [])
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

@resource_dependencies @read albedo::Color

@resource_dependencies @write albedo::Color

@resource_dependencies begin
  @read normal::Color
  @write albedo::Color
end

@resource_dependencies begin
  @read normal::Color
  @write
  albedo::Color
  pbr::Color
end
```
"""
macro resource_dependencies(ex)
  reads = []
  writes = []
  ret = Expr(:vect)
  args = @match ex begin
    Expr(:block, _...) => ex.args
    Expr(:macrocall, &Symbol("@read") || &Symbol("@write"), _...) => [ex]
    _ => error("Expected block declaration, got $ex")
  end
  mode = nothing
  for line in args
    isa(line, LineNumberNode) && continue
    if Meta.isexpr(line, :macrocall)
      mode = @match line.args[1] begin
        &Symbol("@read") => :read
        &Symbol("@write") => :write
        m => error("Expected call to @read or @write macro, got a call to macro $m instead")
      end
      mode === :read && length(line.args) > 2 && (append!(reads, line.args[3:end]); mode = nothing)
      mode === :write && length(line.args) > 2 && (append!(writes, line.args[3:end]); mode = nothing)
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
    :($r::Buffer::Vertex) => (r => RESOURCE_USAGE_VERTEX_BUFFER)
    :($r::Buffer::Index) => (r => RESOURCE_USAGE_INDEX_BUFFER)
    :($r::Buffer::Storage) => (r => RESOURCE_USAGE_BUFFER | RESOURCE_USAGE_STORAGE)
    :($r::Buffer::Indirect) => (r => RESOURCE_USAGE_BUFFER | RESOURCE_USAGE_INDIRECT_BUFFER)
    :($r::Buffer::Uniform) => (r => RESOURCE_USAGE_BUFFER | RESOURCE_USAGE_UNIFORM)
    :($r::Buffer::Physical) => (r => RESOURCE_USAGE_PHYSICAL_BUFFER)
    :($r::Buffer) => (r => RESOURCE_USAGE_BUFFER)
    :($r::Color) => (r => RESOURCE_USAGE_COLOR_ATTACHMENT)
    :($r::Depth) => (r => RESOURCE_USAGE_DEPTH_ATTACHMENT)
    :($r::Stencil) => (r => RESOURCE_USAGE_STENCIL_ATTACHMENT)
    :($r::Depth::Stencil) || :($_::Stencil::Depth) => (r => RESOURCE_USAGE_DEPTH_ATTACHMENT | RESOURCE_USAGE_STENCIL_ATTACHMENT)
    :($r::Texture) => (r => RESOURCE_USAGE_TEXTURE)
    :($r::Image::Storage) => (r => RESOURCE_USAGE_IMAGE | RESOURCE_USAGE_STORAGE)
    :($r::Input) => (r => RESOURCE_USAGE_INPUT_ATTACHMENT)
    ::Symbol => error("Resource type annotation required for $ex")
    _ => error("Invalid or unsupported resource type annotation for $ex")
  end
end

function sort_nodes(rg::RenderGraph, node_uses::Dictionary{NodeID, Dictionary{ResourceID, ResourceUsage}})
  collect(rg.nodes)
end

function add_resource_dependencies!(rg::RenderGraph, node::RenderNode)
  get!(Dictionary{ResourceID, Vector{ResourceUsage}}, rg.uses, node.id)
  for command in node.commands
    deps = resource_dependencies(command)
    for (resource_id, resource_dependency) in pairs(deps)
      add_resource_dependency!(rg, node, resource_id, resource_dependency)
    end
  end
end

function add_resource_dependencies!(rg::RenderGraph)
  for node in rg.nodes
    add_resource_dependencies!(rg, node)
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
          resolve_attachment = logical_attachment(attachment.format, attachment.dims, attachment.mip_range, attachment.layer_range, attachment.aspect)
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

function allocate_blocks!(rg::RenderGraph, materialized_resources)
  for node in rg.nodes
    allocate_blocks!(rg, node, materialized_resources)
  end
end

function allocate_blocks!(rg::RenderGraph, node::RenderNode, materialized_resources)
  for command in node.commands
    is_graphics(command) || is_compute(command) || continue
    allocate_block!(command.impl::Union{GraphicsCommand, ComputeCommand}, rg.allocator, rg.device, node.id, materialized_resources)
  end
end

function allocate_block!(command::Union{GraphicsCommand, ComputeCommand}, allocator::LinearAllocator, device::Device, node_id::NodeID, materialized_resources)
  isnothing(command.data) && return command
  data_address = device_address_block!(allocator, device.descriptors, materialized_resources, node_id, command.data)
  command.data_address = data_address
  command
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
