struct Buffer <: LavaAbstraction
  handle::Vk.Buffer
  size::Int64
  offset::Int64
  stride::Int64
  usage_flags::Vk.BufferUsageFlag
  queue_family_indices::Vector{Int8}
  sharing_mode::Vk.SharingMode
  memory::RefValue{Memory}
end

vk_handle_type(::Type{Buffer}) = Vk.Buffer

Vk.bind_buffer_memory(buffer::Buffer, memory::Memory) = Vk.bind_buffer_memory(device(buffer), buffer, memory, memory.offset)

isallocated(buffer::Buffer) = isdefined(buffer.memory, 1)

device_address(buffer::Buffer) = Vk.get_buffer_device_address(device(buffer), Vk.BufferDeviceAddressInfo(handle(buffer))) + UInt64(buffer.offset)

function Buffer(
  device,
  size::Integer;
  usage_flags = Vk.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
  queue_family_indices = queue_family_indices(device),
  sharing_mode = Vk.SHARING_MODE_EXCLUSIVE,
)
  info = Vk.BufferCreateInfo(size, usage_flags, sharing_mode, queue_family_indices)
  handle = unwrap(create(Buffer, device, info))
  buffer = Buffer(handle, size, 0, 0, usage_flags, queue_family_indices, sharing_mode, Ref{Memory}())
end

function Base.show(io::IO, buffer::Buffer)
  print(io, Buffer, "($(Base.format_bytes(buffer.size))), $(buffer.usage_flags)")
  if !isallocated(buffer)
    print(io, ", unallocated")
  end
  print(io, ')')
end

function Base.similar(buffer::Buffer; memory_domain = nothing, usage_flags = buffer.usage_flags)
  similar = Buffer(device(buffer), buffer.size; usage_flags, buffer.queue_family_indices, buffer.sharing_mode)
  if isallocated(buffer)
    memory_domain = @something(memory_domain, buffer.memory[].domain)
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

@inline function Base.view(buffer::Buffer, range::StepRange)
  @boundscheck checkbounds(buffer, range)
  setproperties(buffer, (; offset = range.start, stride = range.step, size = range.stop - range.start))
end

@inline function Base.view(buffer::Buffer, range::UnitRange)
  @boundscheck checkbounds(buffer, range)
  setproperties(buffer, (; offset = range.start, stride = 0, size = range.stop - range.start))
end

Base.checkbounds(buffer::Buffer, range::AbstractRange) = range.stop ≤ buffer.size || throw(BoundsError(buffer, range))

Base.firstindex(buffer::Buffer) = buffer.offset
Base.lastindex(buffer::Buffer) = buffer.size

"""
Allocate memory and bind it to the provided buffer.
"""
function allocate!(buffer::Buffer, domain::MemoryDomain)
  !isallocated(buffer) || error("Can't allocate memory for a buffer more than once.")
  device = Lava.device(buffer)
  reqs = Vk.get_buffer_memory_requirements(device, buffer)
  memory = Memory(device, reqs.size, reqs.memory_type_bits, domain)
  bind!(buffer, memory)
end

function bind!(buffer::Buffer, memory::Memory)
  !isdefined(buffer.memory, 1) || error("Buffers can't be bound to memory twice.")
  unwrap(Vk.bind_buffer_memory(buffer, memory))
  buffer.memory[] = memory
  memory.is_bound[] = true
  buffer
end
