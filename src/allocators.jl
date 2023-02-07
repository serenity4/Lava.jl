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

DeviceAddress(la::LinearAllocator) = DeviceAddress(la.buffer)

Base.copyto!(la::LinearAllocator, data::T, layout::LayoutStrategy = NativeLayout()) where {T} = copyto!(la, serialize(data, layout), T, layout)
Base.copyto!(la::LinearAllocator, bytes, T::DataType, layout::LayoutStrategy) = copyto!(la, bytes, alignment(layout, T))
function Base.copyto!(la::LinearAllocator, bytes::Vector{UInt8}, alignment::Integer)
  offset = get_offset(la, alignment)
  data_size = length(bytes)
  if offset + data_size > la.buffer.size
    error("Data does not fit in memory (available: $(la.buffer.size - offset), requested: $data_size).")
  end
  ptrcopy!(la.base_ptr + offset, bytes)
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
