mutable struct LinearAllocator
    buffer::BufferBlock{MemoryBlock}
    last_offset::Int
    base_ptr::Ptr{Cvoid}
end

device(allocator::LinearAllocator) = device(allocator.buffer)

function LinearAllocator(device, size)
    buffer = BufferBlock(device, size; usage = Vk.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
    allocate!(buffer, MEMORY_DOMAIN_HOST)
    mem = memory(buffer)
    LinearAllocator(buffer, 0, unwrap(map(memory(buffer))))
end

device_address(la::LinearAllocator) = device_address(la.buffer)

Base.copyto!(la::LinearAllocator, data::AbstractArray, alignment = 8) = _copyto!(la, collect(data), alignment)
Base.copyto!(la::LinearAllocator, data::AbstractString, alignment = 8) = _copyto!(la, string(data), alignment)
Base.copyto!(la::LinearAllocator, data, alignment = 8) = _copyto!(la, data, alignment)

function _copyto!(la::LinearAllocator, data, alignment) where {T}
    offset = get_offset(la, data, alignment)
    ptrcopy!(la.base_ptr + offset, data)
    size = sizeof(data)
    la.last_offset = offset + size
    @view la.buffer[offset:offset + size]
end

function get_offset(la::LinearAllocator, data, alignment)
    offset = alignment * cld(la.last_offset, alignment)
    offset + sizeof(data) ≤ size(la.buffer) || error("Data does not fit in memory (available: $(size(la.buffer) - offset), requested: $(sizeof(data))).")
    offset
end

function reset!(la::LinearAllocator)
    la.last_offset = 0
end
