mutable struct LinearAllocator
  buffer::BufferBlock{MemoryBlock}
  last_offset::Int
  base_ptr::Ptr{Cvoid}
end

device(allocator::LinearAllocator) = device(allocator.buffer)

function LinearAllocator(device, size)
  buffer = BufferBlock(device, size; usage = Vk.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
  allocate!(buffer, MEMORY_DOMAIN_DEVICE_HOST)
  LinearAllocator(buffer, 0, map(memory(buffer)))
end

device_address(la::LinearAllocator) = device_address(la.buffer)

_collect_data(data::AbstractArray) = collect(data)
_collect_data(data::AbstractString) = string(data)
_collect_data(data) = data
Base.copyto!(la::LinearAllocator, data, alignment = 8) = _copyto!(la, _collect_data(data), alignment)

function _copyto!(la::LinearAllocator, data, alignment) where {T}
  offset = get_offset(la, data, alignment)
  ptrcopy!(la.base_ptr + offset, data)
  size = sizeof(data)
  la.last_offset = offset + size
  @view la.buffer[offset:(offset + size)]
end

function get_offset(la::LinearAllocator, data, alignment)
  offset = alignment * cld(la.last_offset, alignment)
  offset + sizeof(data) ≤ size(la.buffer) ||
    error("Data does not fit in memory (available: $(size(la.buffer) - offset), requested: $(sizeof(data))).")
  offset
end

function reset!(la::LinearAllocator)
  la.last_offset = 0
end
