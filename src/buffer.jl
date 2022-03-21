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
  size::Int64
  usage::Vk.BufferUsageFlag
  queue_family_indices::Vector{Int8}
  sharing_mode::Vk.SharingMode
  memory::RefValue{M}
end

Base.size(buffer::BufferBlock) = buffer.size
usage(buffer::BufferBlock) = buffer.usage

device_address(buffer::BufferBlock) = Vk.get_buffer_device_address(device(buffer), Vk.BufferDeviceAddressInfo(handle(buffer)))

function BufferBlock(
  device,
  size;
  usage = Vk.BufferUsageFlag(0),
  queue_family_indices = queue_family_indices(device),
  sharing_mode = Vk.SHARING_MODE_EXCLUSIVE,
  memory_type = MemoryBlock,
  allocate = false,
)
  usage |= Vk.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
  info = Vk.BufferCreateInfo(size, usage, sharing_mode, queue_family_indices)
  handle = unwrap(create(BufferBlock, device, info))
  buffer = BufferBlock(handle, size, usage, convert(Vector{Int8}, queue_family_indices), sharing_mode, Ref{memory_type}())
end

function Base.show(io::IO, block::BufferBlock)
  print(io, BufferBlock, "($(Base.format_bytes(size(block))), $(block.usage)")
  if !isallocated(block)
    print(io, ", unallocated")
  end
  print(io, ')')
end

function Base.similar(buffer::BufferBlock{T}; memory_domain = nothing, usage = buffer.usage) where {T}
  similar = BufferBlock(device(buffer), size(buffer); usage, buffer.queue_family_indices, buffer.sharing_mode, memory_type = T)
  if isallocated(buffer)
    memory_domain = @something(memory_domain, memory(buffer).domain)
    allocate!(similar, memory_domain)
  end
  similar
end

function ptrcopy!(ptr, data::DenseArray{T}) where {T}
  check_isbits(T)
  GC.@preserve data unsafe_copyto!(Ptr{T}(ptr), pointer(data), length(data))
end

function ptrcopy!(ptr, data::T) where {T}
  check_isbits(T)
  ref = Ref{T}(data)
  GC.@preserve ref unsafe_copyto!(Ptr{T}(ptr), Base.unsafe_convert(Ptr{T}, ref), 1)
end

function ptrcopy!(ptr, data::String)
  GC.@preserve data unsafe_copyto!(Ptr{UInt8}(ptr), Base.unsafe_convert(Ptr{UInt8}, data), sizeof(data))
end

function check_isbits(@nospecialize(T))
  isbitstype(T) || error("Expected isbits type for a pointer copy operation, got data of type $T")
end

struct SubBuffer{B<:DenseBuffer} <: Buffer{SubMemory}
  buffer::B
  offset::Int64
  stride::Int64
  size::Int64
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
function allocate!(buffer::DenseBuffer, domain::MemoryDomain)
  _device = device(buffer)
  reqs = Vk.get_buffer_memory_requirements(_device, buffer)
  memory = MemoryBlock(_device, reqs.size, reqs.memory_type_bits, domain)
  bind!(buffer, memory)
end

function bind!(buffer::BufferBlock, memory::Memory)
  unwrap(Vk.bind_buffer_memory(buffer, memory))
  buffer.memory[] = memory
  memory.is_bound[] = true
  buffer
end
