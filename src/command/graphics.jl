abstract type DrawCommand end

@struct_hash_equal struct RenderTargets
  color::Vector{Resource}
  depth::Optional{Resource}
  stencil::Optional{Resource}
  function RenderTargets(color::AbstractVector, depth, stencil)
    check_dimensions(color, depth, stencil)
    new(color, depth, stencil)
  end
end

function check_dimensions(color, depth, stencil)
  all_dimensions = Vector{Int64}[]
  for r in [color; depth; stencil]
    isnothing(r) && continue
    dims = dimensions(r)
    !isnothing(dims) && push!(all_dimensions, dims)
  end
  isempty(all_dimensions) && return
  dim, dims = all_dimensions[1], @view all_dimensions[2:end]
  all(==(dim), dims) || throw(ArgumentError("Color, depth and/or stencil attachments have inconsistent dimensions."))
end

RenderTargets(color::AbstractVector; depth = nothing, stencil = nothing) = RenderTargets(color, depth, stencil)
RenderTargets(color...; depth = nothing, stencil = nothing) = RenderTargets(collect(color); depth, stencil)

function Vk.PipelineRenderingCreateInfo(targets::RenderTargets)
  color_formats = [image_format(c) for c in targets.color]
  depth_format = isnothing(targets.depth) ? Vk.FORMAT_UNDEFINED : image_format(targets.depth)
  stencil_format = isnothing(targets.stencil) ? Vk.FORMAT_UNDEFINED : image_format(targets.stencil)
  Vk.PipelineRenderingCreateInfo(0, color_formats, depth_format, stencil_format)
end

mutable struct GraphicsCommand <: CommandImplementation
  const draw::DrawCommand
  const program::Program
  const data::Optional{ProgramInvocationData}
  data_address::DeviceAddressBlock
  const targets::RenderTargets
  const state::DrawState
  const resource_dependencies::Dictionary{Resource, ResourceDependency}
end

GraphicsCommand(draw::DrawCommand, program::Program, data::ProgramInvocationData, targets::RenderTargets, draw_state::DrawState, resource_dependencies::Dictionary = Dictionary{Resource, ResourceDependency}()) =
  GraphicsCommand(draw, program, data, DeviceAddressBlock(0), targets, draw_state, resource_dependencies)
GraphicsCommand(draw::DrawCommand, program::Program, data_address::DeviceAddressBlock, targets::RenderTargets, draw_state::DrawState, resource_dependencies::Dictionary = Dictionary{Resource, ResourceDependency}()) =
  GraphicsCommand(draw, program, nothing, data_address, targets, draw_state, resource_dependencies)
GraphicsCommand(draw::DrawCommand, program::Program, data::Union{ProgramInvocationData, DeviceAddressBlock}, color::Resource...; depth = nothing, stencil = nothing, render_state = RenderState(), invocation_state = ProgramInvocationState(), resource_dependencies::Dictionary = Dictionary{Resource, ResourceDependency}()) = GraphicsCommand(draw, program, data, RenderTargets(color...; depth, stencil), DrawState(render_state, invocation_state), resource_dependencies)
GraphicsCommand(draw::DrawCommand, program::Program, data::Union{ProgramInvocationData, DeviceAddressBlock}, targets::RenderTargets, render_state::RenderState, invocation_state::ProgramInvocationState, resource_dependencies::Dictionary = Dictionary{Resource, ResourceDependency}()) = GraphicsCommand(draw, program, data, targets, DrawState(render_state, invocation_state), resource_dependencies)

resource_dependencies(command::GraphicsCommand) = command.resource_dependencies

function deduce_render_area(command::GraphicsCommand)
  (; targets) = command
  !isempty(targets.color) || error("Cannot deduce the render area for `GraphicsCommand` with no color attachments.")
  RenderArea(attachment_dimensions(targets.color[1])...)
end

struct DrawIndirect <: DrawCommand
  parameters::Resource
  count::Int64
end

function apply(cb::CommandBuffer, draw::DrawIndirect, resources)
  (; buffer) = get_physical_resource(resources, draw.parameters)
  Vk.cmd_draw_indirect(cb, buffer, buffer.offset, draw.count, buffer.stride)
end

"""
Indexed draw.

Indices start from 1 to the number of vertices used for the draw call.
These are vertex indices, which will later be allocated contiguously inside a single [`IndexData`](@ref) buffer.
The corresponding buffer range will be kept in memory for each indexed draw in the `indices` field of `IndexData`.
"""
struct DrawIndexed <: DrawCommand
  vertex_offset::Int32
  indices::Vector{UInt32}
  instances::UnitRange{Int64}
end

DrawIndexed(indices; instances = 1:1, vertex_offset::Integer = -1) = DrawIndexed(vertex_offset, indices, instances)

struct IndexData
  index_list::Vector{UInt32}
  index_buffer::RefValue{Buffer}
  indices::IdDict{DrawIndexed,UnitRange{Int64}}
end

IndexData() = IndexData(UInt32[], Ref{Buffer}(), IdDict())

"Append new indices to `idata`, returning the corresponding range of indices to be used for indexed draw calls."
function Base.append!(id::IndexData, command::DrawIndexed)
  start = lastindex(id.index_list) + 1
  stop = lastindex(id.index_list) + length(command.indices)
  append!(id.index_list, command.indices)
  id.indices[command] = start:stop
  nothing
end

function apply(cb::CommandBuffer, draw::DrawIndexed, index_data::IndexData)
  indices = index_data.indices[draw]
  Vk.cmd_draw_indexed(
    cb,
    length(indices),
    length(draw.instances),
    indices.start - 1,
    draw.vertex_offset,
    draw.instances.start - 1,
  )
end

struct DrawIndexedIndirect <: DrawCommand
  parameters::Resource
  count::Int64
end

function apply(cb::CommandBuffer, draw::DrawIndexedIndirect, resources)
  (; buffer) = get_physical_resource(draw.parameters)
  Vk.cmd_draw_indexed_indirect(cb, buffer, buffer.offset, draw.count, buffer.stride)
end
