struct IndexData
  index_list::Vector{UInt32}
  index_buffer::RefValue{BufferBlock{MemoryBlock}}
end

IndexData() = IndexData(UInt32[], Ref{BufferBlock{MemoryBlock}}())

function allocate_index_buffer(id::IndexData, device::Device)
  #TODO: Create index buffer in render graph to avoid excessive synchronization.
  id.index_buffer[] = buffer(device, id.index_list .- 1U; usage = Vk.BUFFER_USAGE_INDEX_BUFFER_BIT)
end

"Append new indices to `idata`, returning the corresponding range of indices to be used for indexed draw calls."
function Base.append!(id::IndexData, idata)
  first_index = lastindex(id.index_list) + 1
  append!(id.index_list, idata)
  first_index:lastindex(id.index_list)
end
