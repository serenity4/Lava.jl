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

"Run finalizers for handles we know may be destroyed once the descriptor is freed."
function run_finalizers(descriptor::Descriptor)
  isa(descriptor.written_state, Vk.DescriptorImageInfo) || return
  finalize(descriptor.written_state.sampler)
end

Vk.DescriptorType(descriptor::Descriptor) = Vk.DescriptorType(descriptor.id)
descriptor_type(id::DescriptorID) = DescriptorType(UInt8(UInt128(id) >> 120))
descriptor_type(descriptor::Descriptor) = descriptor_type(descriptor.id)

function assert_type(descriptor::Descriptor, dtype::DescriptorType)
  @assert descriptor_type(descriptor) == dtype "Descriptor type is $(descriptor_type(descriptor)) (expected $dtype)"
  descriptor
end

storage_image_descriptor(resource::Resource, node = nothing) = Descriptor(DESCRIPTOR_TYPE_STORAGE_IMAGE, resource, node)
sampler_descriptor(sampler::Sampling, node = nothing) = Descriptor(DESCRIPTOR_TYPE_SAMPLER, sampler, node)
sampled_image_descriptor(resource::Resource, node = nothing) = Descriptor(DESCRIPTOR_TYPE_SAMPLED_IMAGE, resource, node)
texture_descriptor(texture::Texture, node = nothing) = Descriptor(DESCRIPTOR_TYPE_TEXTURE, texture, node)

struct DescriptorArray
  descriptors::Dictionary{UInt32,DescriptorID}
  indices::Dictionary{DescriptorID,UInt32}
  holes::Vector{UInt32}
end

DescriptorArray() = DescriptorArray(Dictionary(), Dictionary(), UInt32[])

function get_descriptor_index!(arr::DescriptorArray, id::DescriptorID)
  existing = get(arr.indices, id, nothing)
  !isnothing(existing) && return existing
  index = isempty(arr.holes) ? UInt32(1 + length(arr.descriptors)) : pop!(arr.holes)
  insert!(arr.descriptors, index, id)
  set!(arr.indices, id, index)
  index
end

function delete_descriptor!(arr::DescriptorArray, id::DescriptorID)
  index = get(arr.indices, id, nothing)
  isnothing(index) && return
  delete!(arr.indices, id)
  unset!(arr.descriptors, index)
  push!(arr.holes, index)
end

struct DescriptorSet <: LavaAbstraction
  handle::Vk.DescriptorSet
  layout::Vk.DescriptorSetLayout
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
  empty!(gdescs.pending)
  @atomic gdescs.counter = 1
  nothing
end

Base.@kwdef struct GlobalDescriptorsConfig
  sampled_images::Int64 = 2048
  storage_images::Int64 = 512
  samplers::Int64 = 2048
  textures::Int64 = 2048
end

const GLOBAL_DESCRIPTOR_SET_INDEX = 0

const BINDING_SAMPLED_IMAGE = 0
const BINDING_STORAGE_IMAGE = 1
const BINDING_SAMPLER = 2
const BINDING_COMBINED_IMAGE_SAMPLER = 3

function GlobalDescriptors(device, config::GlobalDescriptorsConfig = GlobalDescriptorsConfig())
  # MEMORY LEAK: This pool is never reset, and so its memory is never reclaimed.
  # Inserting a naive finalizer produces a segfault, needs another solution.
  pool = Vk.DescriptorPool(
    device,
    1,
    [
      Vk.DescriptorPoolSize(Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE, config.sampled_images),
      Vk.DescriptorPoolSize(Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE, config.storage_images),
      Vk.DescriptorPoolSize(Vk.DESCRIPTOR_TYPE_SAMPLER, config.samplers),
      Vk.DescriptorPoolSize(Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, config.textures),
    ],
  )

  layout = Vk.DescriptorSetLayout(device,
    [
      Vk.DescriptorSetLayoutBinding(BINDING_SAMPLED_IMAGE, Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE, Vk.SHADER_STAGE_ALL; descriptor_count = config.sampled_images),
      Vk.DescriptorSetLayoutBinding(BINDING_STORAGE_IMAGE, Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE, Vk.SHADER_STAGE_ALL; descriptor_count = config.storage_images),
      Vk.DescriptorSetLayoutBinding(BINDING_SAMPLER, Vk.DESCRIPTOR_TYPE_SAMPLER, Vk.SHADER_STAGE_ALL; descriptor_count = config.samplers),
      Vk.DescriptorSetLayoutBinding(BINDING_COMBINED_IMAGE_SAMPLER, Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, Vk.SHADER_STAGE_ALL; descriptor_count = config.textures),
    ], next = Vk.DescriptorSetLayoutBindingFlagsCreateInfo(repeat([Vk.DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT], 4)))

  set = DescriptorSet(first(unwrap(Vk.allocate_descriptor_sets(device, Vk.DescriptorSetAllocateInfo(pool, [layout])))), layout)
  GlobalDescriptors(pool, set, Dictionary(), Dictionary(), Dictionary(), 0)
end

# Must only be called in-between cycles.
function free_descriptor_batch!(gdescs::GlobalDescriptors, batch::Int64)
  !haskey(gdescs.pending, batch) && return
  for id in gdescs.pending[batch]
    dtype = Vk.DescriptorType(id)
    descriptor = gdescs.descriptors[id]
    delete_descriptor!(gdescs.arrays[dtype], id)
    delete!(gdescs.descriptors, id)
    run_finalizers(descriptor)
  end
  delete!(gdescs.pending, batch)
  nothing
end

"""
Refers to a 0-based Vulkan descriptor index, which may be used to index into arrays of descriptors.

Relies on the Vulkan feature `descriptor_indexing`, which allows such indices to be used flexibly.
"""
primitive type DescriptorIndex 32 end

DescriptorIndex(index::UInt32) = reinterpret(DescriptorIndex, index)
DescriptorIndex(index::Integer) = DescriptorIndex(convert(UInt32, index))

Base.convert(::Type{DescriptorIndex}, idx::UInt32) = reinterpret(DescriptorIndex, idx)
Base.convert(::Type{DescriptorIndex}, idx::Integer) = convert(DescriptorIndex, convert(UInt32, idx))
Base.convert(::Type{UInt32}, idx::DescriptorIndex) = reinterpret(UInt32, idx)

Base.getindex(arr::AbstractVector, idx::DescriptorIndex) = getindex(arr, convert(UInt32, idx))
Base.getindex(arr::Arr, idx::DescriptorIndex) = getindex(arr, convert(UInt32, idx))

Base.show(io::IO, desc::DescriptorIndex) = print(io, DescriptorIndex, '(', reinterpret(UInt32, desc), ')')

SPIRV.primitive_type_to_spirv(::Type{DescriptorIndex}) = UInt32
