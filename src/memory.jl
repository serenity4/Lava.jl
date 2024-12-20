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

struct Memory <: LavaAbstraction
  handle::Vk.DeviceMemory
  offset::Int64
  size::Int64
  property_flags::Vk.MemoryPropertyFlag
  domain::MemoryDomain
  is_bound::RefValue{Bool}
  host_ptr::RefValue{Ptr{Cvoid}}
end

vk_handle_type(::Type{Memory}) = Vk.DeviceMemory

ismapped(memory::Memory) = memory.host_ptr[] ≠ C_NULL
isbound(memory::Memory) = memory.is_bound[]

function Base.map(memory::Memory, size::Integer = memory.size, offset::Integer = memory.offset)
  if Vk.MEMORY_PROPERTY_HOST_COHERENT_BIT ∉ memory.property_flags
    unwrap(Vk.invalidate_mapped_memory_ranges(device(memory), [Vk.MappedMemoryRange(C_NULL, memory, offset, size)]))
  end
  ptr = unwrap(Vk.map_memory(device(memory), memory, offset, size))
  memory.host_ptr[] = ptr
  ptr
end

function unmap(memory::Memory)
  if Vk.MEMORY_PROPERTY_HOST_COHERENT_BIT ∉ memory.property_flags
    unwrap(Vk.flush_mapped_memory_ranges(device(memory), [Vk.MappedMemoryRange(C_NULL, memory, memory.offset, memory.size)]))
  end
  memory.host_ptr[] = C_NULL
  Vk.unmap_memory(device(memory), memory)
end

function Base.map(f, memory::Memory)
  if ismapped(memory)
    f(memory.host_ptr[])
  else
    ptr = map(memory)
    ret = f(ptr)
    unmap(memory)
    ret
  end
end

struct OutOfDeviceMemoryError <: Exception
  requested_size::Int64
end

function Base.showerror(io::IO, exc::OutOfDeviceMemoryError)
  print(io, "OutOfDeviceMemoryError")
  iszero(exc.requested_size) && return
  print(io, " (requested allocation size: ", Base.format_bytes(exc.requested_size), ')')
end

function Memory(device, size::Integer, type::Integer, domain::MemoryDomain)
  ret = allocate_memory(device, size, type, domain)
  !iserror(ret) && return unwrap(ret)
  (; code) = unwrap_error(ret)
  # Try to free up some memory if the error is memory-related, otherwise just throw via unwrapping.
  in(code, (Vk.ERROR_OUT_OF_DEVICE_MEMORY, Vk.ERROR_OUT_OF_HOST_MEMORY)) || return unwrap(ret)
  for full in (false, true)
    GC.gc(full)
    ret = allocate_memory(device, size, type, domain)
    !iserror(ret) && return unwrap(ret)
  end
  (; code) = unwrap_error(ret)
  if code == Vk.ERROR_OUT_OF_DEVICE_MEMORY
    throw(OutOfDeviceMemoryError(size))
  elseif code == Vk.ERROR_OUT_OF_HOST_MEMORY
    throw(OutOfMemoryError())
  end
end

function allocate_memory(device, size, type, domain, flags = Vk.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)::ResultTypes.Result{Memory, Vk.VulkanError}
  prop, i = find_memory_type(physical_device(device), type, domain)
  next = Vk.MemoryAllocateFlagsInfo(0; flags)
  @propagate_errors memory = create(Memory, device, Vk.MemoryAllocateInfo(size, i; next))
  Memory(memory, 0, size, prop.property_flags, domain, Ref(false), Ref(C_NULL))
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
  n = mem_props.memory_type_count
  memtypes = mem_props.memory_types[1:n]
  candidate_indices = findall(i -> type_flag & (1 << (i - 1)) ≠ 0, 1:n)
  index = argmax(i -> f(memtypes[i].property_flags), candidate_indices)
  memtypes[index], index - 1
end

function memory_view(memory::Memory, offset, size)
  size > 0 || error("Size must be positive")
  offset + size ≤ memory.size || error("A memory view cannot extend beyond its underlying memory")
  setproperties(memory, (; offset, size))
end

@inline function Base.view(memory::Memory, range::UnitRange)
  @boundscheck checkbounds(memory, range)
  memory_view(memory, range.start, range.stop - range.start)
end

Base.checkbounds(memory::Memory, range::UnitRange) = range.stop ≤ memory.size || throw(BoundsError(memory, range))

Base.firstindex(memory::Memory) = memory.offset
Base.lastindex(memory::Memory) = memory.size

function Base.show(io::IO, memory::Memory)
  print(io, Memory, "($(Base.format_bytes(memory.size)), $(memory.domain))")
end
