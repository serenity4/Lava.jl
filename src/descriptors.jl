primitive type DescriptorID 128 end

DescriptorID(id::UInt128) = reinterpret(DescriptorID, id)
Base.UInt128(id::DescriptorID) = reinterpret(UInt128, id)
DescriptorID(id::UUID) = DescriptorID(UInt128(id))

@enum DescriptorType::UInt8 begin
  DESCRIPTOR_TYPE_STORAGE_IMAGE
  DESCRIPTOR_TYPE_SAMPLER
  DESCRIPTOR_TYPE_SAMPLED_IMAGE
  DESCRIPTOR_TYPE_TEXTURE
  DESCRIPTOR_TYPE_ATTACHMENT
  DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE
end

function DescriptorID(type::DescriptorType)
  id = UInt128(uuid())
  DescriptorID((id << 8) >> 8 + (UInt128(type) << 120))
end

Vk.DescriptorType(id::DescriptorID) = Vk.DescriptorType(descriptor_type(id))
function Vk.DescriptorType(t::DescriptorType)
  @match t begin
    &DESCRIPTOR_TYPE_STORAGE_IMAGE => Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE
    &DESCRIPTOR_TYPE_SAMPLER => Vk.DESCRIPTOR_TYPE_SAMPLER
    &DESCRIPTOR_TYPE_SAMPLED_IMAGE => Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE
    &DESCRIPTOR_TYPE_TEXTURE => Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
    &DESCRIPTOR_TYPE_ATTACHMENT => Vk.DESCRIPTOR_TYPE_INPUT_ATTACHMENT
    &DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE => Vk.DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
  end
end

# Not sure whether this is relevant.
@bitmask DescriptorFlags::UInt32 begin
  DESCRIPTOR_IS_LOGICAL = 1
end

struct Descriptor
  id::DescriptorID
  data::Union{Resource,Sampling,Texture}
  flags::DescriptorFlags
  node_id::Optional{NodeID}
  written_state::Union{Nothing,Vk.DescriptorImageInfo,Vk.DescriptorBufferInfo}
end

Descriptor(type::DescriptorType, data, node_id::Optional{NodeID} = nothing; flags = DescriptorFlags(0)) = Descriptor(DescriptorID(type), data, flags, node_id, nothing)

Vk.DescriptorType(descriptor::Descriptor) = Vk.DescriptorType(descriptor.id)
descriptor_type(id::DescriptorID) = DescriptorType(UInt8(UInt128(id) >> 120))
descriptor_type(descriptor::Descriptor) = descriptor_type(descriptor.id)

assert_type(descriptor::Descriptor, dtype::DescriptorType) = @assert descriptor_type(descriptor) == dtype "Descriptor type is $(descriptor_type(descriptor)) (expected $dtype)"

storage_image_descriptor(image::Resource, node = nothing) = Descriptor(DESCRIPTOR_TYPE_STORAGE_IMAGE, image, node)
sampler_descriptor(sampler::Sampling, node = nothing) = Descriptor(DESCRIPTOR_TYPE_SAMPLER, sampler, node)
sampled_image_descriptor(image::Resource, node = nothing) = Descriptor(DESCRIPTOR_TYPE_SAMPLED_IMAGE, image, node)
texture_descriptor(tex::Texture, node = nothing) = Descriptor(DESCRIPTOR_TYPE_TEXTURE, tex, node)

struct DescriptorArray
  descriptors::Dictionary{UInt32,DescriptorID}
  indices::Dictionary{DescriptorID,UInt32}
  holes::Vector{UInt32}
end

DescriptorArray() = DescriptorArray(Dictionary(), Dictionary(), UInt32[])

function new_descriptor!(arr::DescriptorArray, id::DescriptorID)
  existing = get(arr.indices, id, nothing)
  !isnothing(existing) && return existing
  index = isempty(arr.holes) ? UInt32(length(arr.descriptors)) : pop!(arr.holes)
  insert!(arr.descriptors, index, id)
  set!(arr.indices, id, index)
  index
end

function delete_descriptor!(arr::DescriptorArray, id::DescriptorID)
  index = arr.indices[id]
  delete!(arr.indices, id)
  delete!(arr.descriptors, index)
  push!(arr.holes, index)
end

mutable struct GlobalDescriptors
  const pool::Vk.DescriptorPool
  const gset::DescriptorSet
  const arrays::Dictionary{Vk.DescriptorType,DescriptorArray}
  const descriptors::Dictionary{DescriptorID, Descriptor}
  const pending::Dictionary{Int64,Vector{DescriptorID}}
  @atomic counter::Int64
end

function Base.delete!(gdescs::GlobalDescriptors, id::DescriptorID)
  descriptor = get(gdescs.descriptors, id, nothing)
  isnothing(descriptor) && return
  delete!(gdescs.descriptors, id)
end

function Base.empty!(gdescs::GlobalDescriptors)
  empty!(gdescs.arrays)
  empty!(gdescs.descriptors)
  nothing
end

Base.@kwdef struct GlobalDescriptorsConfig
  sampled_images::Int64 = 2048
  storage_images::Int64 = 512
  samplers::Int64 = 2048
  textures::Int64 = 2048
end

function GlobalDescriptors(device, config::GlobalDescriptorsConfig = GlobalDescriptorsConfig())
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
      Vk.DescriptorSetLayoutBinding(0, Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE, Vk.SHADER_STAGE_ALL; descriptor_count = config.sampled_images),
      Vk.DescriptorSetLayoutBinding(1, Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE, Vk.SHADER_STAGE_ALL; descriptor_count = config.storage_images),
      Vk.DescriptorSetLayoutBinding(2, Vk.DESCRIPTOR_TYPE_SAMPLER, Vk.SHADER_STAGE_ALL; descriptor_count = config.samplers),
      Vk.DescriptorSetLayoutBinding(3, Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, Vk.SHADER_STAGE_ALL; descriptor_count = config.textures),
    ], next = Vk.DescriptorSetLayoutBindingFlagsCreateInfo(repeat([Vk.DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT], 4)))

  set = DescriptorSet(first(unwrap(Vk.allocate_descriptor_sets(device, Vk.DescriptorSetAllocateInfo(pool, [layout])))), layout)
  GlobalDescriptors(pool, set, Dictionary(), Dictionary(), Dictionary(), 0)
end

# Must only be called in-between cycles.
function free_descriptor_batch!(gdescs::GlobalDescriptors, batch::Int64)
  !haskey(gdescs.pending, batch) && return
  for id in gdescs.pending[batch]
    dtype = Vk.DescriptorType(id)
    delete_descriptor!(gdescs.arrays[dtype], id)
    delete!(gdescs.descriptors, id)
  end
  delete!(gdescs.pending, batch)
end

primitive type DescriptorIndex 32 end

DescriptorIndex(index::UInt32) = reinterpret(DescriptorIndex, index)
DescriptorIndex(index::Integer) = DescriptorIndex(convert(UInt32, index))

Base.convert(::Type{DescriptorIndex}, idx::UInt32) = reinterpret(DescriptorIndex, idx)
Base.convert(::Type{DescriptorIndex}, idx::Integer) = convert(DescriptorIndex, convert(UInt32, idx))
Base.convert(::Type{UInt32}, idx::DescriptorIndex) = reinterpret(UInt32, idx)

Base.getindex(arr::Union{Arr,AbstractVector}, idx::DescriptorIndex) = getindex(arr, convert(UInt32, idx))

Base.show(io::IO, desc::DescriptorIndex) = print(io, DescriptorIndex, '(', reinterpret(UInt32, desc), ')')

SPIRV.primitive_type_to_spirv(::Type{DescriptorIndex}) = SPIRV.IntegerType(32, 0)
