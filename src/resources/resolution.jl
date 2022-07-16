"""
Request a 0-based descriptor index for use in a shader.
"""
function request_descriptor_index end

function request_descriptor_index(rg::RenderGraph, node::RenderNode, texture::Texture)
  (; descriptors) = rg
  id = uuid()
  idx = new_descriptor!(get!(DescriptorArray, descriptors.arrays, Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER), id)
  insert!(descriptors.render_nodes, id, uuid(node))
  insert!(descriptors.textures, id, texture)
  idx
end

function PhysicalDescriptors(device::Device, uses::Dictionary{NodeUUID,ResourceUses}, resources::PhysicalResources, descriptors::LogicalDescriptors)
  res = PhysicalDescriptors(device)
  (; sampled_images, storage_images, samplers, textures) = res.gset
  for (dtype, arr) in pairs(descriptors.arrays)
    @switch dtype begin
      @case &Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE
      for (id, index) in pairs(arr.indices)
        (; image) = descriptors.storage_images[id]
        isa(image, LogicalImage) && (image = resources[image])
        sampler = Vk.Sampler(device, sampling)
        layout = image_layout(uses[descriptors.render_nodes[id]].images[uuid(image)])
        insert!(res.gset.storage_images, index, Vk.DescriptorImageInfo(empty_handle(Vk.Sampler), default_view(image), Vk.IMAGE_LAYOUT_UNDEFINED))
      end

      @case &Vk.DESCRIPTOR_TYPE_SAMPLER
      for (id, index) in pairs(arr.indices)
        sampling = descriptors.samplers[id]
        sampler = Vk.Sampler(device, sampling)
        insert!(res.gset.samplers, index, Vk.DescriptorImageInfo(sampler, empty_handle(Vk.ImageView), Vk.IMAGE_LAYOUT_UNDEFINED))
      end

      @case &Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
      for (id, index) in pairs(arr.indices)
        texture = descriptors.textures[id]
        sampler = Vk.Sampler(device, texture.sampling)
        (; image) = texture
        isa(image, LogicalImage) && (image = resources[image])
        layout = image_layout(uses[descriptors.render_nodes[id]].images[uuid(image)])
        insert!(res.gset.textures, index, Vk.DescriptorImageInfo(sampler, default_view(image), layout))
      end
    end
  end
  res
end
