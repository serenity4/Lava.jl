abstract type Memory <: LavaAbstraction end

offset(memory::Memory) = 0

abstract type DenseMemory <: Memory end

"""
Memory domains:
- `MEMORY_DOMAIN_HOST` is host-visible memory, ideal for uploads to the GPU. It is preferably coherent and non-cached.
- `MEMORY_DOMAIN_HOST_CACHED` is host-visible, cached memory, ideal for readbacks from the GPU. It is preferably coherent.
- `MEMORY_DOMAIN_DEVICE` is device-local. It may be visible (integrated GPUs).
"""
@enum MemoryDomain::Int8 begin
  MEMORY_DOMAIN_HOST
  MEMORY_DOMAIN_HOST_CACHED
  MEMORY_DOMAIN_DEVICE
  MEMORY_DOMAIN_DEVICE_HOST
end

struct MemoryBlock <: DenseMemory
  handle::Vk.DeviceMemory
  size::Int
  properties::Vk.MemoryPropertyFlag
  domain::MemoryDomain
  is_bound::Ref{Bool}
  ptr::Ref{Ptr{Cvoid}}
end

vk_handle_type(::Type{MemoryBlock}) = Vk.DeviceMemory

Base.size(block::MemoryBlock) = block.size

properties(block::MemoryBlock) = block.properties

ismapped(block::MemoryBlock) = block.ptr[] ≠ C_NULL
isbound(block::MemoryBlock) = block.is_bound[]

function Base.map(memory::DenseMemory, size::Integer = size(memory), offset::Integer = offset(memory))
  if Vk.MEMORY_PROPERTY_HOST_COHERENT_BIT ∉ properties(memory)
    unwrap(Vk.invalidate_mapped_memory_ranges(device(memory), [Vk.MappedMemoryRange(C_NULL, memory, offset, size)]))
  end
  ptr = unwrap(Vk.map_memory(device(memory), memory, offset, size))
  memory.ptr[] = ptr
  ptr
end

function unmap(memory::DenseMemory)
  if Vk.MEMORY_PROPERTY_HOST_COHERENT_BIT ∉ properties(memory)
    unwrap(Vk.flush_mapped_memory_ranges(device(memory), [Vk.MappedMemoryRange(C_NULL, memory, offset(memory), size(memory))]))
  end
  memory.ptr[] = C_NULL
  Vk.unmap_memory(device(memory), memory)
end

function Base.map(f, memory::DenseMemory)
  if ismapped(memory)
    f(memory.ptr[])
  else
    ptr = map(memory)
    ret = f(ptr)
    unmap(memory)
    ret
  end
end

MemoryBlock(device, size::Integer, type::Integer, domain::MemoryDomain) = unwrap(memory_block(device, size, type, domain))

function memory_block(device, size, type, domain)::ResultTypes.Result{MemoryBlock, Vk.VulkanError}
  prop, i = find_memory_type(physical_device(device), type, domain)
  next = Vk.MemoryAllocateFlagsInfo(0; flags = Vk.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)
  @propagate_errors memory = create(MemoryBlock, device, Vk.MemoryAllocateInfo(size, i; next))
  MemoryBlock(memory, size, prop.property_flags, domain, Ref(false), Ref(C_NULL))
end

find_memory_type(physical_device, type_flag, domain::MemoryDomain) = find_memory_type(Base.Fix1(score, domain), physical_device, type_flag)

function score(domain::MemoryDomain, properties)
  @match domain begin
    &MEMORY_DOMAIN_HOST =>
      (10 * (Vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT in properties)) +
      (Vk.MEMORY_PROPERTY_HOST_COHERENT_BIT in properties) -
      (Vk.MEMORY_PROPERTY_HOST_CACHED_BIT in properties)
    &MEMORY_DOMAIN_HOST_CACHED =>
      (10 * (Vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT | Vk.MEMORY_PROPERTY_HOST_CACHED_BIT in properties)) +
      (Vk.MEMORY_PROPERTY_HOST_COHERENT_BIT in properties)
    &MEMORY_DOMAIN_DEVICE =>
      (Vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT in properties) - 2 * (Vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT in properties)
    &MEMORY_DOMAIN_DEVICE_HOST =>
      (Vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT in properties) + 2 * (Vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT in properties)
  end
end

function find_memory_type(f, physical_device, type_flag)
  mem_props = Vk.get_physical_device_memory_properties(physical_device)
  memtypes = mem_props.memory_types[1:(mem_props.memory_type_count)]
  candidate_indices = findall(i -> type_flag & (1 << i) ≠ 0, 0:(length(memtypes) - 1))
  index = argmax(i -> f(memtypes[i].property_flags), candidate_indices)
  memtypes[index], index - 1
end

function find_memory_type(physical_device, type_flag, properties::Vk.MemoryPropertyFlag)
  mem_props = Vk.get_physical_device_memory_properties(physical_device)
  indices = findall(x -> properties in x.property_flags, mem_props.memory_types[1:(mem_props.memory_type_count)]) .- 1
  if isempty(indices)
    error("Could not find memory with properties $properties")
  else
    ind = findfirst(i -> type_flag & (1 << i) ≠ 0, indices)
    if isnothing(ind)
      error("Could not find memory with type $type_flag")
    else
      indices[ind], mem_props[indices[ind]]
    end
  end
end

struct SubMemory{M<:DenseMemory} <: DenseMemory
  memory::M
  offset::Int
  size::Int
end

function SubMemory(memory, size; offset = 0)
  size > 0 || error("Size must be positive")
  offset + size > Lava.size(memory) || error("A SubMemory cannot extend beyond its underlying memory")
  SubMemory(memory, offset, size)
end

handle(sub::SubMemory) = handle(sub.memory)

offset(sub::SubMemory) = sub.offset

Base.size(sub::SubMemory) = sub.size

function Base.view(memory::DenseMemory, range::UnitRange)
  range_check(memory, range)
  SubMemory(memory, range.start, range.stop - range.start)
end

range_check(domain, range) = range.stop ≤ size(domain) || throw(BoundsError(domain, range))

Base.firstindex(memory::Memory) = offset(memory)
Base.lastindex(memory::Memory) = size(memory)

"Memory that can't be accessed by the renderer."
struct OpaqueMemory <: Memory end

function Base.show(io::IO, block::MemoryBlock)
  print(io, MemoryBlock, "($(Base.format_bytes(size(block))), $(block.domain))")
end
