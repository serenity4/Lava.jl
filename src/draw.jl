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
  Vk.cmd_draw_indirect(cb, buffer, offset(buffer), draw.count, stride(buffer))
end

"""
Indexed draw.

Indices start from 1 to the number of vertices used for the draw call.
These are vertex indices, which will later be allocated contiguously inside a single [`IndexData`](@ref) buffer.
The corresponding buffer range will be kept in memory for each indexed draw in the `indices` field of `IndexData`.
"""
struct DrawIndexed <: DrawCommand
  vertex_offset::Int64
  indices::Vector{Int64}
  instances::UnitRange{Int64}
end

struct IndexData
  index_list::Vector{UInt32}
  index_buffer::RefValue{BufferBlock{MemoryBlock}}
  indices::IdDict{DrawIndexed,UnitRange{Int64}}
end

IndexData() = IndexData(UInt32[], Ref{BufferBlock{MemoryBlock}}(), IdDict())

function allocate_index_buffer(id::IndexData, device::Device)
  #TODO: Create index buffer in render graph to avoid excessive synchronization.
  id.index_buffer[] = buffer(device, id.index_list .- 1U; usage = Vk.BUFFER_USAGE_INDEX_BUFFER_BIT)
end

"Append new indices to `idata`, returning the corresponding range of indices to be used for indexed draw calls."
function Base.append!(id::IndexData, command::DrawIndexed)
  first_index = lastindex(id.index_list) + 1
  append!(id.index_list, command.indices)
  range = first_index:lastindex(id.index_list)
  id.indices[command] = range
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
  Vk.cmd_draw_indexed_indirect(cb, buffer, offset(buffer), draw.count, stride(buffer))
end
