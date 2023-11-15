"""
Request a 0-based descriptor index for use in a shader.

The index will only be valid until the end of the next render operation.
"""
function request_index!(gdescs::GlobalDescriptors, descriptor::Descriptor)
  idx = new_descriptor!(get!(DescriptorArray, gdescs.arrays, Vk.DescriptorType(descriptor)), descriptor.id)
  if haskey(gdescs.descriptors, descriptor.id)
    gdescs.descriptors[descriptor.id] === descriptor || error(descriptor.id, " has already been registered for a different descriptor.")
  else
    insert!(gdescs.descriptors, descriptor.id, descriptor)
  end
  idx
end
request_index!(device::Device, descriptor::Descriptor) = request_index!(device.descriptors, descriptor)

function write_descriptors!(gdescs::GlobalDescriptors, descriptors, uses::Dictionary{NodeID,Dictionary{ResourceID,ResourceUsage}}, resources::Dictionary{ResourceID, Resource})
  (; gset) = gdescs
  device = Lava.device(gset)
  writes = Vk.WriteDescriptorSet[]
  batch_ids = DescriptorID[]

  for descriptor in descriptors
    haskey(uses, descriptor.node_id::NodeID) || continue
    type = descriptor_type(descriptor)
    dtype = Vk.DescriptorType(type)
    @switch type begin
      @case &DESCRIPTOR_TYPE_SAMPLED_IMAGE
      resource = descriptor.data::Resource
      islogical(resource) && (resource = resources[resource.id])
      usage = uses[descriptor.node_id::NodeID][resource.id].usage::ImageUsage
      layout = image_layout(usage.type, usage.access)
      info = Vk.DescriptorImageInfo(empty_handle(Vk.Sampler), default_view(resource.data::Image), layout)
      new_descriptor = @set descriptor.written_state = info
      gdescs.descriptors[descriptor.id] = new_descriptor
      index = gdescs.arrays[dtype].indices[descriptor.id]
      push!(writes, Vk.WriteDescriptorSet(gset.handle, 0, index - 1, dtype, [info], [], []))

      @case &DESCRIPTOR_TYPE_STORAGE_IMAGE
      resource = descriptor.data::Resource
      islogical(resource) && (resource = resources[resource.id])
      usage = uses[descriptor.node_id::NodeID][resource.id].usage::ImageUsage
      layout = image_layout(usage.type, usage.access)
      info = Vk.DescriptorImageInfo(empty_handle(Vk.Sampler), default_view(resource.data::Image), layout)
      new_descriptor = @set descriptor.written_state = info
      gdescs.descriptors[descriptor.id] = new_descriptor
      index = gdescs.arrays[dtype].indices[descriptor.id]
      push!(writes, Vk.WriteDescriptorSet(gset.handle, 1, index - 1, dtype, [info], [], []))

      @case &DESCRIPTOR_TYPE_SAMPLER
      sampling = descriptor.data::Sampling
      sampler = Vk.Sampler(device, sampling)
      info = Vk.DescriptorImageInfo(empty_handle(Vk.Sampler), empty_handle(Vk.ImageView), Vk.IMAGE_LAYOUT_UNDEFINED)
      new_descriptor = @set descriptor.written_state = info
      gdescs.descriptors[descriptor.id] = new_descriptor
      index = gdescs.arrays[dtype].indices[descriptor.id]
      push!(writes, Vk.WriteDescriptorSet(gset.handle, 2, index - 1, dtype, [info], [], []))

      @case &DESCRIPTOR_TYPE_TEXTURE
      texture = descriptor.data::Texture
      sampler = Vk.Sampler(device, texture.sampling)
      (; image) = texture
      islogical(image) && (texture = @set texture.image = resources[image.id])
      usage = uses[descriptor.node_id::NodeID][image.id].usage::ImageUsage
      layout = image_layout(usage.type, usage.access)
      info = Vk.DescriptorImageInfo(sampler, ImageView(image.data::Image), layout)
      new_descriptor = @set descriptor.written_state = info
      gdescs.descriptors[descriptor.id] = new_descriptor
      index = gdescs.arrays[dtype].indices[descriptor.id]
      push!(writes, Vk.WriteDescriptorSet(gset.handle, 3, index - 1, dtype, [info], [], []))
    end

    push!(batch_ids, descriptor.id)
  end

  Vk.update_descriptor_sets(device, writes, [])
  batch_index = (@atomic gdescs.counter += 1)
  insert!(gdescs.pending, batch_index, batch_ids)
  batch_index
end
