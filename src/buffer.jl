"""
Buffer backed by memory of type `M`.

Offsets, strides and sizes are always expressed in bytes.
"""
abstract type Buffer{M<:Memory} <: LavaAbstraction end

memory_type(::Type{<:Buffer{M}}) where {M} = M
memory_type(t) = memory_type(typeof(t))

vk_handle_type(::Type{<:Buffer}) = Vk.Buffer

Vk.bind_buffer_memory(buffer::Buffer, memory::Memory) = Vk.bind_buffer_memory(device(buffer), buffer, memory, offset(memory))

# abstract type DataOperation end

# abstract type Transfer <: DataOperation end
# abstract type VideoDecode <: DataOperation end
# abstract type VideoEncode <: DataOperation end

# abstract type StorageType end

# abstract type Vertex <: StorageType end
# abstract type Index{T} <: StorageType end

# abstract type SparsityType end

# abstract type SparseResidency <: SparsityType end
# abstract type SparseBinding <: SparsityType end

# abstract type Sparse{T<:SparsityType,O<:LavaAbstraction} <: StorageType end

abstract type DenseBuffer{M<:DenseMemory} <: Buffer{M} end

isallocated(buffer::DenseBuffer) = isdefined(buffer.memory, 1)
memory(buffer::DenseBuffer) = buffer.memory[]

struct BufferBlock{M<:DenseMemory} <: DenseBuffer{M}
    handle::Vk.Buffer
    size::Int
    usage::Vk.BufferUsageFlag
    queue_family_indices::Vector{Int8}
    sharing_mode::Vk.SharingMode
    memory::Ref{M}
end

Base.size(buffer::BufferBlock) = buffer.size

device_address(buffer::BufferBlock) = Vk.get_buffer_device_address(device(buffer), Vk.BufferDeviceAddressInfo(handle(buffer)))

function BufferBlock(device, size; usage = Vk.BufferUsageFlag(0), queue_family_indices = queue_family_indices(device), sharing_mode = Vk.SHARING_MODE_EXCLUSIVE, memory_type = MemoryBlock)
    usage |= Vk.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
    info = Vk.BufferCreateInfo(size, usage, sharing_mode, queue_family_indices)
    handle = unwrap(create(BufferBlock, device, info))
    buffer = BufferBlock(handle, size, usage, convert(Vector{Int8}, queue_family_indices), sharing_mode, Ref{memory_type}())
end

function buffer(device, data; kwargs...)
    buffer = BufferBlock(device, sizeof(data); kwargs...)
    unwrap(allocate!(buffer, MEMORY_DOMAIN_HOST))
    ret = copyto!(buffer, data)
    if !isnothing(ret)
        # there was a staging operation required
        semaphore = ret
        push!(device.staging_ops, semaphore)
    end
    buffer
end

function Base.similar(buffer::BufferBlock{T}, domain::MemoryDomain) where {T}
    similar = BufferBlock(device(buffer), size(buffer); usage = buffer.usage, queue_family_indices = buffer.queue_family_indices, sharing_mode = buffer.sharing_mode, memory_type = T)
    if isallocated(buffer)
        unwrap(allocate!(similar, domain))
    end
    similar
end

function Base.copyto!(buffer::BufferBlock, data)
    mem = memory(buffer)
    if Vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT in properties(mem)
        tmp = similar(buffer, MEMORY_DOMAIN_HOST)
        copyto!(tmp, data)
        copyto!(tmp, buffer)
    elseif Vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT in properties(mem)
        ptr = unwrap(map(mem))
        ptrcopy!(ptr, data)
        unmap(mem)
        nothing
    else
        error("Buffer not visible neither to device nor to host (memory properties: $(properties(mem))).")
    end
end

function ptrcopy!(ptr, data::DenseArray{T}) where {T}
    GC.@preserve data unsafe_copyto!(Ptr{T}(ptr), pointer(data), length(data))
end

function ptrcopy!(ptr, data::T) where {T}
    ref = Ref{T}(data)
    GC.@preserve ref unsafe_copyto!(Ptr{T}(ptr), Base.unsafe_convert(Ptr{T}, ref), 1)
end

function ptrcopy!(ptr, data::String)
    GC.@preserve data unsafe_copyto!(Ptr{UInt8}(ptr), Base.unsafe_convert(Ptr{UInt8}, data), sizeof(data))
end

struct SubBuffer{B<:DenseBuffer} <: Buffer{SubMemory}
    buffer::B
    offset::Int
    stride::Int
    size::Int
end

@forward SubBuffer.buffer handle

Base.size(sub::SubBuffer) = sub.size

offset(buffer::Buffer) = 0
offset(buffer::SubBuffer) = buffer.offset

stride(buffer::Buffer) = 0
stride(buffer::SubBuffer) = buffer.stride

memory(sub::SubBuffer) = @view memory(sub.buffer)[offset(sub):(size(sub) - offset(sub))]

device_address(sub::SubBuffer) = device_address(sub.buffer) + UInt64(sub.offset)

function Base.view(buffer::DenseBuffer, range::StepRange)
    range_check(buffer, range)
    SubBuffer(buffer, range.start, range.step, range.stop - range.start)
end

function Base.view(buffer::DenseBuffer, range::UnitRange)
    range_check(buffer, range)
    SubBuffer(buffer, range.start, 0, range.stop - range.start)
end

Base.firstindex(buffer::Buffer) = offset(buffer)
Base.lastindex(buffer::Buffer) = size(buffer)

"""
Allocate a `MemoryBlock` and bind it to the provided buffer.
"""
function allocate!(buffer::DB, domain::MemoryDomain)::Result{DB,Vk.VulkanError} where {DB<:DenseBuffer}
    _device = device(buffer)
    reqs = Vk.get_buffer_memory_requirements(_device, buffer)
    @propagate_errors memory = MemoryBlock(_device, reqs.size, reqs.memory_type_bits, domain)
    @propagate_errors bind!(buffer, memory)
end

function bind!(buffer::BufferBlock, memory::Memory)::Result{BufferBlock,Vk.VulkanError}
    buffer.memory[] = memory
    @propagate_errors Vk.bind_buffer_memory(buffer, memory)
    buffer
end
