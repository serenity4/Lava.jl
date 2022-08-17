mutable struct LinearAllocator
  buffer::Buffer
  last_offset::Int64
  base_ptr::Ptr{Cvoid}
end

available_size(la::LinearAllocator, alignment = 0) = la.buffer.size - get_offset(la, alignment)

device(la::LinearAllocator) = device(la.buffer)

function LinearAllocator(device, size)
  b = Buffer(device; size, usage_flags = Vk.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, memory_domain = MEMORY_DOMAIN_DEVICE_HOST)
  LinearAllocator(b, 0, map(b.memory[]))
end

device_address(la::LinearAllocator) = device_address(la.buffer)

bytes(data::AbstractArray) = Base.reinterpret(UInt8, collect(data))
bytes(data::AbstractString) = Vector{UInt8}(string(data))
function bytes(data::T) where {T}
  check_isbits(T)
  bytes([data])
end

Base.copyto!(la::LinearAllocator, data, alignment = 8) = copyto!(la, bytes(data), alignment)

function Base.copyto!(la::LinearAllocator, data::Vector{UInt8}, alignment)
  offset = get_offset(la, alignment)
  data_size = sizeof(data)
  if offset + data_size > la.buffer.size
    error("Data does not fit in memory (available: $(la.buffer.size - offset), requested: $data_size).")
  end
  ptrcopy!(la.base_ptr + offset, data)
  la.last_offset = offset + data_size
  @view la.buffer[offset:la.last_offset]
end

function get_offset(la::LinearAllocator, alignment)
  iszero(alignment) && return la.last_offset
  alignment * cld(la.last_offset, alignment)
end

function reset!(la::LinearAllocator)
  la.last_offset = 0
end
