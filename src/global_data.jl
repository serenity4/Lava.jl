"""
Frame-global structure that holds all data needed in the frame.

Its linear allocator is used for allocating lots of small objects, like material parameters and vertex data.

Other resources that require a global descriptor set (bind-once strategy) are put into a `ResourceDescriptors`.
This includes general image data & samplers, with the corresponding descriptors.

The index list is used to append index data and is turned into an index buffer before initiating the render sequence.
"""
struct GlobalData
  allocator::LinearAllocator
  resources::ResourceDescriptors
  index_list::Vector{UInt32}
  index_buffer::RefValue{BufferBlock{MemoryBlock}}
end

GlobalData(device) = GlobalData(
  LinearAllocator(device, 1_000_000), # 1 MB
  ResourceDescriptors(device),
  [],
  Ref{BufferBlock{MemoryBlock}}(),
)

function Base.write(gset::GlobalDescriptorSet)
  (; state, set) = gset
  types = [Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE, Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE, Vk.DESCRIPTOR_TYPE_SAMPLER, Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER]
  writes = Vk.WriteDescriptorSet[]
  for (i, type) in enumerate(types)
    infos = state[type]
    !isempty(infos) || continue
    write = Vk.WriteDescriptorSet(set.handle, i - 1, 0, type, infos, [], [])
    push!(writes, write)
  end
  !isempty(writes) || return
  Vk.update_descriptor_sets(device(set), writes, [])
end

function allocate_index_buffer(gd::GlobalData, device::Device)
  #TODO: Create index buffer in render graph to avoid excessive synchronization.
  gd.index_buffer[] = buffer(device, convert(Vector{UInt32}, gd.index_list .- 1); usage = Vk.BUFFER_USAGE_INDEX_BUFFER_BIT)
end
