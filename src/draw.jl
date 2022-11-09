abstract type DrawCommand end

struct DrawDirect <: DrawCommand
  vertices::Vector{Int64}
  instances::Vector{Int64}
end

function apply(cb::CommandBuffer, draw::DrawDirect)
  Vk.cmd_draw(
    cb,
    1 + draw.vertices.stop - draw.vertices.start,
    1 + draw.instances.stop - draw.instances.start,
    draw.vertices.start - 1,
    draw.instances.start - 1,
  )
end

struct DrawIndirect{B<:Buffer} <: DrawCommand
  parameters::B
  count::Int64
end

function apply(cb::CommandBuffer, draw::DrawIndirect)
  buffer = draw.parameters
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

function allocate_index_buffer(id::IndexData, device::Device)
  #TODO: Create index buffer in render graph to avoid excessive synchronization.
  id.index_buffer[] = Buffer(device; data = id.index_list, usage_flags = Vk.BUFFER_USAGE_INDEX_BUFFER_BIT)
end

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
    1 + indices.stop - indices.start,
    1 + draw.instances.stop - draw.instances.start,
    indices.start - 1,
    draw.vertex_offset,
    draw.instances.start - 1,
  )
end

struct DrawIndexedIndirect{B<:Buffer} <: DrawCommand
  parameters::B
  count::Int64
end

function apply(cb::CommandBuffer, draw::DrawIndexedIndirect)
  buffer = draw.parameters
  Vk.cmd_draw_indexed_indirect(cb, buffer, buffer.offset, draw.count, buffer.stride)
end

@auto_hash_equals struct RenderTargets
  color::Vector{Resource}
  depth::Optional{Resource}
  stencil::Optional{Resource}
end

RenderTargets(color::AbstractVector; depth = nothing, stencil = nothing) = RenderTargets(color, depth, stencil)
RenderTargets(color...; depth = nothing, stencil = nothing) = RenderTargets(collect(color); depth, stencil)

function Vk.PipelineRenderingCreateInfo(targets::RenderTargets)
  color_formats = [(c.data::Attachment).view.format for c in targets.color]
  depth_format = isnothing(targets.depth) ? Vk.FORMAT_UNDEFINED : (targets.depth.data::Attachment).view.format
  stencil_format = isnothing(targets.stencil) ? Vk.FORMAT_UNDEFINED : (targets.stencil.data::Attachment).view.format
  Vk.PipelineRenderingCreateInfo(0, color_formats, depth_format, stencil_format)
end
