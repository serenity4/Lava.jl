"""
Buffer backed by memory of type `M`.
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

size(buffer::BufferBlock) = buffer.size

function BufferBlock(device, size, usage; queue_family_indices = queue_family_indices(device), sharing_mode = Vk.SHARING_MODE_EXCLUSIVE, memory_type = MemoryBlock)
    info = Vk.BufferCreateInfo(size, usage, sharing_mode, queue_family_indices)
    handle = unwrap(create(BufferBlock, device, info))
    BufferBlock(handle, size, usage, convert(Vector{Int8}, queue_family_indices), sharing_mode, Ref{memory_type}())
end

struct SubBuffer{B<:DenseBuffer} <: Buffer{SubMemory}
    buffer::B
    offset::Int
    stride::Int
    size::Int
end

@forward SubBuffer.buffer handle

size(sub::SubBuffer) = sub.size

offset(buffer::Buffer) = 0
offset(buffer::SubBuffer) = buffer.offset

stride(buffer::Buffer) = 0
stride(buffer::SubBuffer) = buffer.stride

memory(sub::SubBuffer) = @view memory(sub.buffer)[offset(sub):(size(sub) - offset(sub))]

SubBuffer(buffer::DenseBuffer; offset = 0, stride = 0) = SubBuffer(buffer, offset, stride)

function Base.view(buffer::DenseBuffer, range::StepRange)
    range.stop ≤ size(buffer) || throw(BoundsError(buffer, range))
    SubBuffer(buffer, range.start, range.step, range.stop - range.start)
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
