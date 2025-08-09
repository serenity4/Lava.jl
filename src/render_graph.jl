contains_fragment_stage(stages::Vk.PipelineStageFlag2) = in(Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, stages) || in(Vk.PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, stages) || in(Vk.PIPELINE_STAGE_2_ALL_COMMANDS_BIT, stages)

Base.@kwdef struct RenderNode
  id::NodeID = NodeID()
  render_area::Optional{RenderArea} = nothing
  commands::Optional{Vector{Command}} = Command[]
  clears::Dictionary{Resource, ClearValue} = Dictionary{Resource, ClearValue}()
  name::Optional{Symbol} = nothing
end

RenderNode(name::Symbol) = RenderNode(; name)
RenderNode(render_area::RenderArea, name = nothing) = RenderNode(; render_area, name)
RenderNode(render_area::Tuple, name = nothing) = RenderNode(RenderArea(render_area), name)

function RenderNode(command::Command, name = nothing)
  render_area = nothing
  is_graphics(command) && (render_area = deduce_render_area(command.graphics))
  RenderNode(; render_area, commands = [command], name)
end
function RenderNode(commands, name = nothing)
  render_area = nothing
  for command in commands
    if is_graphics(command)
      if isnothing(render_area)
        render_area = deduce_render_area(command.graphics)
      else
        render_area_2 = deduce_render_area(command.graphics)
        render_area == render_area_2 || error("Render areas are not identical across commands: $render_area ≠ $render_area_2")
      end
    end
  end
  RenderNode(; render_area, commands, name)
end
Base.convert(::Type{RenderNode}, command::Command) = RenderNode(command)

function update_commands!(node::RenderNode, added, deleted)
  splice!(node.commands, findall(in(deleted), node.commands))
  append!(node.commands, added)
  node
end

print_name(io::IO, node::RenderNode) = printstyled(IOContext(io, :color => true), isnothing(node.name) ? node.id : node.name; color = 210)

Descriptor(type::DescriptorType, data, node::RenderNode; flags = DescriptorFlags(0)) = Descriptor(type, data, node.id; flags)

function ResourceUsage(resource::Resource, node::RenderNode, dep::ResourceDependency)
  usage_flags = resource_usage_flags(resource, node, dep)
  stages = stage_flags(resource, node, dep)
  usage = @match resource_type(resource) begin
    &RESOURCE_TYPE_BUFFER => BufferUsage(; dep.type, dep.access, stages, usage_flags)
    &RESOURCE_TYPE_IMAGE || &RESOURCE_TYPE_IMAGE_VIEW => ImageUsage(; dep.type, dep.access, stages, usage_flags, dep.samples)
    &RESOURCE_TYPE_ATTACHMENT => AttachmentUsage(;
        dep.type,
        dep.access,
        dep.clear_value,
        dep.samples,
        stages,
        usage_flags,
        aspect = @something(aspect_flags(dep), aspect_flags(resource.data::Union{LogicalAttachment, Attachment})),
      )
  end
  ResourceUsage(resource.id, dep.type, usage)
end

function resource_usage_flags(resource::Resource, node::RenderNode, dep::ResourceDependency)
  isbuffer(resource) && return buffer_usage_flags(dep.type, dep.access)
  flags = image_usage_flags(dep.type, dep.access)
  haskey(node.clears, resource) && (flags |= Vk.IMAGE_USAGE_TRANSFER_DST_BIT)
  return flags
end

function stage_flags(resource::Resource, node::RenderNode, dependency::ResourceDependency)
  flags = command_stage_flags(node, resource)
  return flags | clear_stage_flags(node, resource, dependency)
end

function command_stage_flags(node::RenderNode, resource::Resource)
  flags = foldl(|, command_stage_flags(command, resource) for command in node.commands; init = Vk.PIPELINE_STAGE_2_NONE)
  return flags
end

function clear_stage_flags(node::RenderNode, resource::Resource, dependency::ResourceDependency)
  haskey(node.clears, resource) || return Vk.PIPELINE_STAGE_2_NONE
  flags = Vk.PIPELINE_STAGE_2_NONE

  dependency.type == RESOURCE_USAGE_COLOR_ATTACHMENT && (flags |= Vk.PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)
  dependency.type == RESOURCE_USAGE_IMAGE && (flags |= Vk.PIPELINE_STAGE_2_CLEAR_BIT)
  dependency.type == RESOURCE_USAGE_TEXTURE && (flags |= Vk.PIPELINE_STAGE_2_CLEAR_BIT)

  in(RESOURCE_USAGE_DEPTH_ATTACHMENT, dependency.type) && (flags |= Vk.PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | Vk.PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)
  in(RESOURCE_USAGE_STENCIL_ATTACHMENT, dependency.type) && (flags |= Vk.PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | Vk.PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)

  @assert !iszero(flags)
  return flags
end

function command_stage_flags(command::Command, resource::Resource)
  flags = Vk.PIPELINE_STAGE_2_NONE
  @match command.impl begin
    graphics::GraphicsCommand => begin
      isattachment(resource) && (flags |= stage_flags(graphics.targets, graphics.state.render_state, resource))
      @trymatch graphics.draw begin
        draw::DrawIndirect => draw.parameters.id == resource.id && (flags |= Vk.PIPELINE_STAGE_2_DRAW_INDIRECT_BIT)
      end
      # XXX: refine further down from Vk.PIPELINE_STAGE_2_ALL_GRAPHICS_BIT
      graphics.data !== nothing && is_resource_used_by_descriptors(resource, graphics.data) && (flags |= Vk.PIPELINE_STAGE_2_ALL_GRAPHICS_BIT)
      dependency = get(graphics.resource_dependencies, resource, nothing)
      dependency !== nothing && (flags |= Vk.PIPELINE_STAGE_2_ALL_GRAPHICS_BIT)
    end
    compute::ComputeCommand => begin
      @trymatch compute.dispatch begin
        dispatch::DispatchIndirect => dispatch.buffer.id == resource.id && (flags |= Vk.PIPELINE_STAGE_2_DRAW_INDIRECT_BIT)
      end
      compute.data !== nothing && is_resource_used_by_descriptors(resource, compute.data) && (flags |= Vk.PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
      dependency = get(compute.resource_dependencies, resource, nothing)
      dependency !== nothing && (flags |= Vk.PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
    end
    transfer::TransferCommand => begin
      if any(r.id == resource.id for r in (transfer.src, transfer.dst))
        is_copy(transfer) && (flags |= Vk.PIPELINE_STAGE_2_COPY_BIT)
        is_resolve(transfer) && (flags |= Vk.PIPELINE_STAGE_2_RESOLVE_BIT)
        is_blit(transfer) && (flags |= Vk.PIPELINE_STAGE_2_BLIT_BIT)
      end
    end
    present::PresentCommand => (flags |= Vk.PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT)
  end
  return flags
end

function is_resource_used_by_descriptors(resource::Resource, data::ProgramInvocationData)
  for descriptor in data.descriptors
    is_resource_used_by_descriptor(resource.id, descriptor) && return true
  end
  return false
end

function is_resource_used_by_descriptor(id::ResourceID, descriptor::Descriptor)
  return @match descriptor.data begin
    resource::Resource => resource.id == id
    sampling::Sampling => false
    texture::Texture => texture.resource.id == id
  end
end

function stage_flags(targets::RenderTargets, render_state::RenderState, resource::Resource)
  stages = Vk.PIPELINE_STAGE_2_NONE
  (; color, depth, stencil) = targets
  any(att.id == resource.id for att in color) && (stages |= Vk.PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)
  depth_stencil_stages = Vk.PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | Vk.PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
  depth !== nothing && resource.id == depth.id && render_state.enable_depth_testing && (stages |= depth_stencil_stages)
  stencil !== nothing && resource.id == stencil.id && render_state.enable_stencil_testing && (stages |= depth_stencil_stages)
  stages
end

function aspect_flags(dependency::ResourceDependency)
  flags = Vk.ImageAspectFlag()
  (; type) = dependency
  in(RESOURCE_USAGE_COLOR_ATTACHMENT, type) && (flags |= Vk.IMAGE_ASPECT_COLOR_BIT)
  in(RESOURCE_USAGE_DEPTH_ATTACHMENT, type) && (flags |= Vk.IMAGE_ASPECT_DEPTH_BIT)
  in(RESOURCE_USAGE_STENCIL_ATTACHMENT, type) && (flags |= Vk.IMAGE_ASPECT_STENCIL_BIT)
  iszero(flags) ? nothing : flags
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
mutable struct RenderGraph
  const device::Device
  "Used to allocate lots of tiny objects."
  const allocator::LinearAllocator
  const resource_graph::SimpleDiGraph{Int64}
  const nodes::Dictionary{NodeID,RenderNode}
  const node_indices::Dictionary{NodeID,Int64}
  const node_indices_inv::Dictionary{Int64,NodeID}
  const resource_indices::Dictionary{ResourceID,Int64}
  "Resource uses per node. One resource may be used several times by a single node."
  const uses::Dictionary{NodeID,Dictionary{ResourceID, Vector{ResourceUsage}}}
  const resources::Dictionary{ResourceID, Resource}

  # State for execution.
  "Pairs att1 => att2 where att1 is a multisampled attachment resolved on att2."
  const resolve_pairs::Dictionary{Resource, Resource}
  const combined_resource_uses::Dictionary{ResourceID, ResourceUsage}
  const combined_node_uses::Dictionary{NodeID,Dictionary{ResourceID, ResourceUsage}}
  const materialized_resources::Dictionary{ResourceID, Resource}
  descriptor_batch_index::Int64
  const index_data::IndexData
end

function RenderGraph(device::Device, nodes = nothing; linear_allocator_size = 2^20 #= 1_000_000 KiB =#)
  rg = RenderGraph(device, LinearAllocator(device, linear_allocator_size), SimpleDiGraph(), Dictionary(), Dictionary(), Dictionary(), Dictionary(), Dictionary(), Dictionary(), Dictionary(), Dictionary(), Dictionary(), Dictionary(), -1, IndexData())
  !isnothing(nodes) && add_nodes!(rg, nodes)
  finalizer(finish!, rg)
end

device(rg::RenderGraph) = rg.device

function add_node!(rg::RenderGraph, node::Union{RenderNode, Command})
  node = convert(RenderNode, node)
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

add_nodes!(rg::RenderGraph, nodes::Union{RenderNode, Command}...) = add_nodes!(rg, collect(nodes))
add_node!(rg::RenderGraph, nodes::Vector{Command}) = add_nodes!(rg, nodes)

function add_nodes!(rg::RenderGraph, nodes)
  for node in nodes
    add_node!(rg, node)
  end
end

function add_resource!(rg::RenderGraph, resource::Resource)
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
  resource_uses = get!(Dictionary{ResourceID,Vector{ResourceUsage}}, rg.uses, node.id)
  uses = get!(Vector{ResourceUsage}, resource_uses, resource.id)
  usage = ResourceUsage(resource, node, dependency)
  combine_with_existing!(uses, usage)
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

"""
Declare resource dependencies for a [`RenderGraph`](@ref) to figure out any required synchronization logic.
"""
function resource_dependencies end

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
      samples = something(samples, read_deps[4], Some(nothing))
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
    push!(ret.args, :($(esc(expr)) => ResourceDependency($(esc.(dependency)...))))
  end
  :(dictionary($ret))
end

function extract_special_usage(ex)
  clear_value = nothing
  samples = nothing
  if Meta.isexpr(ex, :call) && ex.args[1] == :(=>)
    clear_value = :($ClearValue($(ex.args[3])))
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

function add_resource_dependencies!(rg::RenderGraph, node::RenderNode)
  get!(Dictionary{ResourceID, Vector{ResourceUsage}}, rg.uses, node.id)
  for command in node.commands
    deps = resource_dependencies(command)
    for (resource, dependency) in pairs(deps)
      add_resource_dependency!(rg, node, resource, dependency)
    end
  end
  for (resource, clear) in pairs(node.clears)
    type = infer_type_for_cleared_resource(resource)
    dependency = ResourceDependency(type, WRITE, clear, nothing)
    add_resource_dependency!(rg, node, resource, dependency)
  end
end

function infer_type_for_cleared_resource(resource::Resource)
  # XXX: We may prefer a specific type of usage for clears instead.
  isimage(resource) && return RESOURCE_USAGE_IMAGE
  aspect = aspect_flags(resource.data)
  in(Vk.IMAGE_ASPECT_COLOR_BIT, aspect) && return RESOURCE_USAGE_COLOR_ATTACHMENT
  usage = ResourceUsageType()
  in(Vk.IMAGE_ASPECT_DEPTH_BIT, aspect) && (usage |= RESOURCE_USAGE_DEPTH_ATTACHMENT)
  in(Vk.IMAGE_ASPECT_STENCIL_BIT, aspect) && (usage |= RESOURCE_USAGE_STENCIL_ATTACHMENT)
  return usage
end

function add_resource_dependencies!(rg::RenderGraph)
  for node in rg.nodes
    add_resource_dependencies!(rg, node)
  end
end

function resolve_attachment_pairs!(rg::RenderGraph)
  for resource in rg.resources
    resource_type(resource) == RESOURCE_TYPE_ATTACHMENT || continue
    haskey(rg.resolve_pairs, resource) && continue
    if islogical(resource)
      resolve_attachment = nothing
      for uses in rg.uses
        resource_uses = get(uses, resource.id, nothing)
        isnothing(resource_uses) && continue
        combined_uses = reduce(merge, resource_uses)
        # XXX: This is wrong, we need to resolve the sample count
        # with all other attachments first instead of falling back to a sample count of 1.
        if samples(resource, combined_uses.usage) > 1
          attachment = resource.data::LogicalAttachment
          resolve_attachment = Resource(LogicalAttachment(attachment.format, attachment.dims, attachment.subresource, 1); name = resolve_attachment_name(resource))
        end
      end
    else
      attachment = resource.data::Attachment
      is_multisampled(attachment) || continue
      resolve_attachment = Resource(LogicalAttachment(attachment.view.format, attachment.view.image.dims, attachment.view.subresource, 1); name = resolve_attachment_name(resource))
    end
    !isnothing(resolve_attachment) && insert!(rg.resolve_pairs, resource, resolve_attachment)
  end
end

resolve_attachment_name(r::Resource) = isnamed(r) ? Symbol(:resolve_, r.name) : nothing

function add_resolve_attachments!(rg::RenderGraph)
  for (resource, resolve_resource) in pairs(rg.resolve_pairs)
    # Add resource in the render graph.
    add_resource!(rg, resolve_resource)

    # Add resource usage for all nodes used by the destination attachment.
    for j in outneighbors(rg.resource_graph, rg.resource_indices[resource.id])
      uses_by_node = rg.uses[rg.node_indices_inv[j]]
      for use in uses_by_node[resource.id]
        @reset use.id = resolve_resource.id
        @reset use.usage.samples = 1
        uses = get!(Vector{ResourceUsage}, uses_by_node, resolve_resource.id)
        combine_with_existing!(uses, use)
      end
    end
  end
end

function combine_with_existing!(uses, use::ResourceUsage)
  @assert all(x -> x.id === use.id, uses)
  i = findfirst(x -> x.type === use.type, uses)
  i === nothing && return push!(uses, use)
  uses[i] = combine(uses[i], use)
end

function materialize_logical_resources!(rg::RenderGraph)
  for use in rg.combined_resource_uses
    resource = rg.resources[use.id]
    islogical(resource) || continue
    existing = get(rg.materialized_resources, resource.id, nothing)
    if !isnothing(existing)
      ret = verify_physical_resource_compatibility_for_use(existing, use)
      !iserror(ret) && continue # we can reuse the resource that has already been materialized
      delete!(rg.materialized_resources, resource.id)
      free(existing)
    end

    @switch resource_type(resource) begin
      @case &RESOURCE_TYPE_BUFFER
      usage = use.usage::BufferUsage
      (; logical_buffer) = resource
      insert!(rg.materialized_resources, resource.id, promote_to_physical(resource, Buffer(rg.device; logical_buffer.size, usage.usage_flags)))

      @case &RESOURCE_TYPE_IMAGE
      usage = use.usage::ImageUsage
      (; logical_image) = resource
      samples = @__MODULE__().samples(resource, usage)
      insert!(rg.materialized_resources, resource.id, promote_to_physical(resource, Image(rg.device; logical_image.format, logical_image.dims, usage.usage_flags, logical_image.layers, logical_image.mip_levels, samples)))

      @case &RESOURCE_TYPE_IMAGE_VIEW
      usage = use.usage::ImageUsage
      samples = @__MODULE__().samples(resource, usage)
      (; logical_image_view) = resource
      (; image, subresource) = logical_image_view
      image_format = logical_image_view.image.format
      insert!(rg.materialized_resources, resource.id, promote_to_physical(resource, ImageView(rg.device; image_format = image.format, image.dims, usage.usage_flags, image.layers, image.mip_levels, samples, view_format = logical_image_view.format, subresource.layer_range, subresource.mip_range)))

      @case &RESOURCE_TYPE_ATTACHMENT
      usage = use.usage::AttachmentUsage
      samples = @__MODULE__().samples(resource, usage)
      (; logical_attachment) = resource
      (; dims) = logical_attachment
      if isnothing(dims)
        # Try to inherit image dimensions from a render area used by a node that also uses this resource.
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
      insert!(rg.materialized_resources, resource.id, promote_to_physical(resource, Attachment(rg.device; logical_attachment.format, dims, samples, usage.aspect, usage.access, usage.usage_flags, logical_attachment.subresource.layer_range, logical_attachment.subresource.mip_range)))
    end
  end
end

function check_physical_resources(rg::RenderGraph)
  for use in rg.combined_resource_uses
    resource = rg.resources[use.id]
    isphysical(resource) || continue
    ret = verify_physical_resource_compatibility_for_use(resource, use)
    iserror(ret) && throw(unwrap_error(ret))
  end
end

struct ResourceNotCompatibleForUse <: Exception
  resource::Resource
  usage_requirements::Optional{Any}
  aspect_requirements::Optional{Any}
end

function Base.showerror(io::IO, exc::ResourceNotCompatibleForUse)
  print(io, "Resource not compatible for use: physical ", exc.resource)
  if !isnothing(exc.usage_requirements)
    found, required = exc.usage_requirements::Tuple
    print(io, " with usage ", found, " was provided, but a usage of ", required, " is required.")
  else
    found, required = exc.aspect_requirements::Tuple
    print(io, " with aspect ", found, " was provided, but is used with an aspect of ", required, " is required.")
  end
end

function verify_physical_resource_compatibility_for_use(resource::Resource, use)::Result{Nothing, ResourceNotCompatibleForUse}
  @switch resource_type(resource) begin
      @case &RESOURCE_TYPE_BUFFER
      usage = use.usage::BufferUsage
      (; buffer) = resource
      in(usage.usage_flags, buffer.usage_flags) || return ResourceNotCompatibleForUse(resource, (buffer.usage_flags, usage.usage_flags), nothing)

      @case &RESOURCE_TYPE_IMAGE
      usage = use.usage::ImageUsage
      (; image) = resource
      in(usage.usage_flags, image.usage_flags) || return ResourceNotCompatibleForUse(resource, (image.usage_flags, usage.usage_flags), nothing)

      @case &RESOURCE_TYPE_IMAGE_VIEW
      usage = use.usage::ImageUsage
      (; image) = resource.image_view
      in(usage.usage_flags, image.usage_flags) || return ResourceNotCompatibleForUse(resource, (image.usage_flags, usage.usage_flags), nothing)

      @case &RESOURCE_TYPE_ATTACHMENT
      usage = use.usage::AttachmentUsage
      (; attachment) = resource
      in(usage.aspect, attachment.view.subresource.aspect) || return ResourceNotCompatibleForUse(resource, nothing, (attachment.view.subresource.aspect, usage.aspect))
      in(usage.usage_flags, attachment.view.image.usage_flags) || return ResourceNotCompatibleForUse(resource, (attachment.view.image.usage_flags, usage.usage_flags), nothing)
    end
    nothing
end

function allocate_blocks!(rg::RenderGraph)
  for node in rg.nodes
    allocate_blocks!(rg, node)
  end
end

function allocate_blocks!(rg::RenderGraph, node::RenderNode)
  for command in node.commands
    is_graphics(command) || is_compute(command) || continue
    allocate_block!(command.impl::Union{GraphicsCommand, ComputeCommand}, rg.allocator, rg.device, node.id, rg.materialized_resources)
  end
end

function allocate_block!(command::Union{GraphicsCommand, ComputeCommand}, allocator::LinearAllocator, device::Device, node_id::NodeID, materialized_resources)
  isnothing(command.data) && return command
  data_address = device_address_block!(allocator, device.descriptors, materialized_resources, node_id, command.data)
  command.data_address = data_address
  command
end

function descriptors_for_cycle(rg::RenderGraph)
  descriptors = Descriptor[]
  for node in rg.nodes
    for command in node.commands
      is_graphics(command) || is_compute(command) || continue
      impl = command.impl::Union{GraphicsCommand, ComputeCommand}
      if !isnothing(impl.data)
        for descriptor in impl.data.descriptors
          # We retrieve the descriptor from the device,
          # because that is the one that has been patched
          # with the correct node ID, unlike `impl.data.descriptors`.
          push!(descriptors, rg.device.descriptors.descriptors[descriptor.id])
        end
      end
    end
  end
  unique!(descriptors)
end

function get_physical_resource(rg::RenderGraph, id::ResourceID)
  get_physical_resource(rg, rg.resources[id])
end

function get_physical_resource(rg::RenderGraph, resource::Resource)
  isphysical(resource) && return resource
  rg.materialized_resources[resource.id]
end

function finish!(rg::RenderGraph)
  reset!(rg.allocator)
  rg.descriptor_batch_index == -1 && return rg
  free_descriptor_batch!(rg.device.descriptors, rg.descriptor_batch_index)
  rg.descriptor_batch_index = -1
  rg
end
