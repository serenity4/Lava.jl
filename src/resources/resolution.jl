"""
Request a 0-based descriptor index for use in a shader.

If the shader resource is a logical resource, the index will only be valid until the end of the next render operation.
"""
function request_descriptor_index end

request_descriptor_index(rg::RenderGraph, node::RenderNode, texture::Texture) = request_descriptor_index(rg.device.logical_descriptors, uuid(node), texture)

function request_descriptor_index(descriptors::LogicalDescriptors, node::NodeUUID, texture::Texture)
  id = uuid()
  idx = new_descriptor!(get!(DescriptorArray, descriptors.arrays, Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER), id)
  insert!(descriptors.textures, id, texture)
  insert!(descriptors.render_nodes, id, node)
  insert!(descriptors.descriptor_types, id, Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
  idx
end

materialize_logical_descriptors!(device::Device, resources::PhysicalResources, uses::Dictionary{NodeUUID,ResourceUses}) = materialize_logical_descriptors!(device.descriptors, device.logical_descriptors, resources, uses)

function materialize_logical_descriptors!(descriptors::PhysicalDescriptors, logical_descriptors::LogicalDescriptors, resources::PhysicalResources, uses::Dictionary{NodeUUID,ResourceUses})
  descriptor_ids = DescriptorUUID[]
  (; gset) = descriptors
  device = Lava.device(gset)
  writes = Vk.WriteDescriptorSet[]

  for (descriptor_id, image) in logical_descriptors.sampled_images
    dtype = Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE
    delete!(logical_descriptors.sampled_images, descriptor_id)
    isa(image, LogicalImage) && (image = resources[image])
    node_id = logical_descriptors.render_nodes[descriptor_id]
    layout = image_layout(uses[node_id].images[uuid(image)])
    info = Vk.DescriptorImageInfo(empty_handle(Vk.Sampler), default_view(image), layout)
    insert!(descriptors.images, descriptor_id, info)
    index = logical_descriptors.arrays[dtype].indices[descriptor_id]
    push!(writes, Vk.WriteDescriptorSet(gset.handle, 0, index, dtype, [info], [], []))
    push!(descriptor_ids, descriptor_id)
  end

  for (descriptor_id, image) in logical_descriptors.storage_images
    dtype = Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE
    delete!(logical_descriptors.storage_images, descriptor_id)
    isa(image, LogicalImage) && (image = resources[image])
    node_id = logical_descriptors.render_nodes[descriptor_id]
    layout = image_layout(uses[node_id].images[uuid(image)])
    info = Vk.DescriptorImageInfo(empty_handle(Vk.Sampler), default_view(image), layout)
    insert!(descriptors.images, descriptor_id, info)
    index = logical_descriptors.arrays[dtype].indices[descriptor_id]
    push!(writes, Vk.WriteDescriptorSet(gset.handle, 1, index, dtype, [info], [], []))
    push!(descriptor_ids, descriptor_id)
  end

  for (descriptor_id, sampling) in pairs(logical_descriptors.samplers)
    dtype = Vk.DESCRIPTOR_TYPE_SAMPLER
    delete!(logical_descriptors.samplers, descriptor_id)
    sampler = Vk.Sampler(device, sampling)
    info = Vk.DescriptorImageInfo(sampler, empty_handle(Vk.ImageView), Vk.IMAGE_LAYOUT_UNDEFINED)
    insert!(descriptors.images, descriptor_id, info)
    index = logical_descriptors.arrays[dtype].indices[descriptor_id]
    push!(writes, Vk.WriteDescriptorSet(gset.handle, 2, index, dtype, [info], [], []))
    push!(descriptor_ids, descriptor_id)
  end

  for (descriptor_id, texture) in pairs(logical_descriptors.textures)
    dtype = Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
    delete!(logical_descriptors.textures, descriptor_id)
    sampler = Vk.Sampler(device, texture.sampling)
    (; image) = texture
    isa(image, LogicalImage) && (image = resources[image])
    node_id = logical_descriptors.render_nodes[descriptor_id]
    layout = image_layout(uses[node_id].images[uuid(image)])
    info = Vk.DescriptorImageInfo(sampler, default_view(image), layout)
    insert!(descriptors.images, descriptor_id, info)
    index = logical_descriptors.arrays[dtype].indices[descriptor_id]
    push!(writes, Vk.WriteDescriptorSet(gset.handle, 3, index, dtype, [info], [], []))
    push!(descriptor_ids, descriptor_id)
  end

  Vk.update_descriptor_sets(device, writes, [])
  descriptor_ids
end

function free_logical_descriptors!(device::Device, descriptor_ids::Vector{DescriptorUUID})
  (; descriptors, logical_descriptors) = device
  for descriptor_id in descriptor_ids
    dtype = logical_descriptors.descriptor_types[descriptor_id]
    delete_descriptor!(logical_descriptors.arrays[dtype], descriptor_id)
    delete!(logical_descriptors.descriptor_types, descriptor_id)
    delete!(logical_descriptors.render_nodes, descriptor_id)
    @switch dtype begin
      @case &Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE || &Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE || &Vk.DESCRIPTOR_TYPE_SAMPLER || &Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
      delete!(descriptors.images, descriptor_id)
    end
  end
end
