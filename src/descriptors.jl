struct DescriptorArray
  descriptors::Dictionary{UInt32,UUID}
  indices::Dictionary{UUID,UInt32}
  holes::Vector{UInt32}
end

DescriptorArray() = DescriptorArray(Dictionary(), Dictionary(), UInt32[])

function new_descriptor!(arr::DescriptorArray, uuid::UUID)
  existing = get(arr.indices, uuid, nothing)
  !isnothing(existing) && return existing
  index = if isempty(arr.holes)
    index = UInt32(length(arr.descriptors))
    insert!(arr.descriptors, index, uuid)
    index
  else
    hole = pop!(arr.holes)
    arr.descriptors[hole] = uuid
    hole
  end
  set!(arr.indices, uuid, index)
  index
end

function delete_descriptor!(arr::DescriptorArray, uuid::UUID)
  index = arr.indices[uuid]
  delete!(arr.indices, uuid)
  delete!(arr.descriptors, index)
  push!(arr.holes, index)
end

Base.@kwdef struct GlobalDescriptorsConfig
  textures::Int64 = 2048
  storage_images::Int64 = 512
  samplers::Int64 = 2048
end

struct DescriptorSetBindingState{T}
  type::Vk.DescriptorType
  elements::Vector{T}
end

struct GlobalDescriptorSet
  set::DescriptorSet
  sampled_images::Dictionary{Int,Vk.DescriptorImageInfo}
  storage_images::Dictionary{Int,Vk.DescriptorImageInfo}
  textures::Dictionary{Int,Vk.DescriptorImageInfo}
  samplers::Dictionary{Int,Vk.DescriptorImageInfo}
end

struct PhysicalDescriptors
  pool::Vk.DescriptorPool
  gset::GlobalDescriptorSet
end

function PhysicalDescriptors(device, config::GlobalDescriptorsConfig = GlobalDescriptorsConfig())
  # MEMORY LEAK: This pool is never reset, and so its memory is never reclaimed.
  # Inserting a naive finalizer produces a segfault, needs another solution.
  pool = Vk.DescriptorPool(
    device,
    1,
    [
      Vk.DescriptorPoolSize(Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1),
      Vk.DescriptorPoolSize(Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE, 1),
      Vk.DescriptorPoolSize(Vk.DESCRIPTOR_TYPE_SAMPLER, 1),
      Vk.DescriptorPoolSize(Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1),
    ],
  )

  layout = Vk.DescriptorSetLayout(device,
    [
      Vk.DescriptorSetLayoutBinding(0, Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE, Vk.SHADER_STAGE_ALL; descriptor_count = config.textures),
      Vk.DescriptorSetLayoutBinding(1, Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE, Vk.SHADER_STAGE_ALL; descriptor_count = config.storage_images),
      Vk.DescriptorSetLayoutBinding(2, Vk.DESCRIPTOR_TYPE_SAMPLER, Vk.SHADER_STAGE_ALL; descriptor_count = config.samplers),
      Vk.DescriptorSetLayoutBinding(3, Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, Vk.SHADER_STAGE_ALL; descriptor_count = config.textures),
    ], next = Vk.DescriptorSetLayoutBindingFlagsCreateInfo(repeat([Vk.DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT], 4)))

  set = DescriptorSet(first(unwrap(Vk.allocate_descriptor_sets(device, Vk.DescriptorSetAllocateInfo(pool, [layout])))), layout)
  gset = GlobalDescriptorSet(set, Dictionary(), Dictionary(), Dictionary(), Dictionary())
  PhysicalDescriptors(pool, gset)
end

function default_view(image::PhysicalImage)
  Vk.ImageView(
    image.image.device,
    image.image,
    image_view_type(length(image.info.dims)),
    format(image),
    Vk.ComponentMapping(
      Vk.COMPONENT_SWIZZLE_IDENTITY,
      Vk.COMPONENT_SWIZZLE_IDENTITY,
      Vk.COMPONENT_SWIZZLE_IDENTITY,
      Vk.COMPONENT_SWIZZLE_IDENTITY,
    ),
    subresource_range(image),
  )
end

function Base.write(gset::GlobalDescriptorSet)
  writes = Vk.WriteDescriptorSet[]
  (; set, textures, samplers, sampled_images, storage_images) = gset
  for (i, info) in pairs(sampled_images)
    write = Vk.WriteDescriptorSet(set.handle, 0, i, Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE, [info], [], [])
    push!(writes, write)
  end
  for (i, info) in pairs(storage_images)
    write = Vk.WriteDescriptorSet(set.handle, 1, i, Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE, [info], [], [])
    push!(writes, write)
  end
  for (i, info) in pairs(samplers)
    write = Vk.WriteDescriptorSet(set.handle, 2, i, Vk.DESCRIPTOR_TYPE_SAMPLER, [info], [], [])
    push!(writes, write)
  end
  for (i, info) in pairs(textures)
    write = Vk.WriteDescriptorSet(set.handle, 3, i, Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, [info], [], [])
    push!(writes, write)
  end
  !isempty(writes) || return
  Vk.update_descriptor_sets(device(set), writes, [])
end
