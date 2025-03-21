"""
Request a 0-based descriptor index for use in a shader.

The index will only be valid until the end of the next render operation.
"""
function request_index!(gdescs::GlobalDescriptors, descriptor::Descriptor)
  array = get!(DescriptorArray, gdescs.arrays, Vk.DescriptorType(descriptor))
  idx = get_descriptor_index!(array, descriptor.id)
  # Make sure the descriptor is made known to the `GlobalDescriptors`, so it is kept during use.
  !haskey(gdescs.descriptors, descriptor.id) && insert!(gdescs.descriptors, descriptor.id, descriptor)
  idx
end
request_index!(device::Device, descriptor::Descriptor) = request_index!(device.descriptors, descriptor)

function write_descriptors!(rg::RenderGraph, descriptors)
  gdescs = rg.device.descriptors
  (; gset) = gdescs
  device = Lava.device(gset)
  writes = Vk.WriteDescriptorSet[]
  descriptor_batch = DescriptorID[]

  for descriptor in descriptors
    haskey(rg.combined_node_uses, descriptor.node_id::NodeID) || continue
    type = descriptor_type(descriptor)
    dtype = Vk.DescriptorType(type)

    @switch type begin
      @case &DESCRIPTOR_TYPE_SAMPLED_IMAGE
      resource = get_physical_resource(rg, descriptor.data::Resource)
      node_uses = rg.combined_node_uses[descriptor.node_id::NodeID]
      usage = get_image_view_usage(resource, node_uses)
      layout = image_layout(usage.type, usage.access)
      image_view = @match resource_type(resource) begin
        &RESOURCE_TYPE_IMAGE_VIEW => resource.image_view
        &RESOURCE_TYPE_IMAGE => ImageView(resource.image)
        type => error("Expected image or image view, got resource of type ", type)
      end
      info = Vk.DescriptorImageInfo(empty_handle(Vk.Sampler), image_view, layout)
      new_descriptor = @set descriptor.written_state = info
      gdescs.descriptors[descriptor.id] = new_descriptor
      index = gdescs.arrays[dtype].indices[descriptor.id]
      push!(writes, Vk.WriteDescriptorSet(gset.handle, BINDING_SAMPLED_IMAGE, index - 1, dtype, [info], [], []))

      @case &DESCRIPTOR_TYPE_STORAGE_IMAGE
      resource = get_physical_resource(rg, descriptor.data::Resource)
      node_uses = rg.combined_node_uses[descriptor.node_id::NodeID]
      usage = get_image_view_usage(resource, node_uses)
      layout = image_layout(usage.type, usage.access)
      image_view = @match resource_type(resource) begin
        &RESOURCE_TYPE_IMAGE_VIEW => resource.image_view
        &RESOURCE_TYPE_IMAGE => ImageView(resource.image)
        type => error("Expected image or image view, got resource of type ", type)
      end
      info = Vk.DescriptorImageInfo(empty_handle(Vk.Sampler), image_view, layout)
      new_descriptor = @set descriptor.written_state = info
      gdescs.descriptors[descriptor.id] = new_descriptor
      index = gdescs.arrays[dtype].indices[descriptor.id]
      push!(writes, Vk.WriteDescriptorSet(gset.handle, BINDING_STORAGE_IMAGE, index - 1, dtype, [info], [], []))

      @case &DESCRIPTOR_TYPE_SAMPLER
      sampling = descriptor.data::Sampling
      sampler = Vk.Sampler(device, sampling)
      info = Vk.DescriptorImageInfo(empty_handle(Vk.Sampler), empty_handle(Vk.ImageView), Vk.IMAGE_LAYOUT_UNDEFINED)
      new_descriptor = @set descriptor.written_state = info
      gdescs.descriptors[descriptor.id] = new_descriptor
      index = gdescs.arrays[dtype].indices[descriptor.id]
      push!(writes, Vk.WriteDescriptorSet(gset.handle, BINDING_SAMPLER, index - 1, dtype, [info], [], []))

      @case &DESCRIPTOR_TYPE_TEXTURE
      texture = descriptor.data::Texture
      sampler = Vk.Sampler(device, texture.sampling)
      resource = get_physical_resource(rg, texture.resource)
      node_uses = rg.combined_node_uses[descriptor.node_id::NodeID]
      usage = get_image_view_usage(resource, node_uses)
      layout = image_layout(usage.type, usage.access)
      info = Vk.DescriptorImageInfo(sampler, resource.image_view, layout)
      new_descriptor = @set descriptor.written_state = info
      gdescs.descriptors[descriptor.id] = new_descriptor
      index = gdescs.arrays[dtype].indices[descriptor.id]
      push!(writes, Vk.WriteDescriptorSet(gset.handle, BINDING_COMBINED_IMAGE_SAMPLER, index - 1, dtype, [info], [], []))
    end

    push!(descriptor_batch, descriptor.id)
  end

  Vk.update_descriptor_sets(device, writes, [])
  batch_index = (@atomic gdescs.counter += 1)
  insert!(gdescs.pending, batch_index, descriptor_batch)
  @assert rg.descriptor_batch_index == -1 "The previous descriptor batch was not freed"
  rg.descriptor_batch_index = batch_index
end

function get_image_view_usage(resource::Resource, node_uses)
  if isimage(resource)
    resource_usage = node_uses[resource.id]
  else
    resource_usage = @something get(node_uses, resource.id, nothing) begin
      image_id = ResourceID(RESOURCE_TYPE_IMAGE, resource.id)
      node_uses[image_id]
    end
  end
  resource_usage.usage::ImageUsage
end
