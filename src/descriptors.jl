const DescriptorUUID = UUID

struct DescriptorArray
  descriptors::Dictionary{UInt32,DescriptorUUID}
  indices::Dictionary{DescriptorUUID,UInt32}
  holes::Vector{UInt32}
end

DescriptorArray() = DescriptorArray(Dictionary(), Dictionary(), UInt32[])

function new_descriptor!(arr::DescriptorArray, uuid::DescriptorUUID)
  existing = get(arr.indices, uuid, nothing)
  !isnothing(existing) && return existing
  index = isempty(arr.holes) ? UInt32(length(arr.descriptors)) : pop!(arr.holes)
  insert!(arr.descriptors, index, uuid)
  set!(arr.indices, uuid, index)
  index
end

function delete_descriptor!(arr::DescriptorArray, uuid::DescriptorUUID)
  index = arr.indices[uuid]
  delete!(arr.indices, uuid)
  delete!(arr.descriptors, index)
  push!(arr.holes, index)
end

struct LogicalDescriptors
  arrays::Dictionary{Vk.DescriptorType,DescriptorArray}
  sampled_images::Dictionary{DescriptorUUID,Union{LogicalImage,PhysicalImage}}
  storage_images::Dictionary{DescriptorUUID,Union{LogicalImage,PhysicalImage}}
  samplers::Dictionary{DescriptorUUID,Sampling}
  textures::Dictionary{DescriptorUUID,Texture}
  render_nodes::Dictionary{DescriptorUUID,NodeUUID}
  descriptor_types::Dictionary{DescriptorUUID, Vk.DescriptorType}
end

LogicalDescriptors() = LogicalDescriptors(Dictionary(), Dictionary(), Dictionary(), Dictionary(), Dictionary(), Dictionary(), Dictionary())

function Base.empty!(logical_descriptors::LogicalDescriptors)
  empty!(logical_descriptors.arrays)
  empty!(logical_descriptors.sampled_images)
  empty!(logical_descriptors.samplers)
  empty!(logical_descriptors. textures)
  empty!(logical_descriptors.render_nodes)
  empty!(logical_descriptors.descriptor_types)
  logical_descriptors
end

Base.@kwdef struct GlobalDescriptorsConfig
  textures::Int64 = 2048
  storage_images::Int64 = 512
  samplers::Int64 = 2048
end

struct PhysicalDescriptors
  pool::Vk.DescriptorPool
  gset::DescriptorSet
  images::Dictionary{DescriptorUUID,Vk.DescriptorImageInfo}
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
  PhysicalDescriptors(pool, set, Dictionary())
end
