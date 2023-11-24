"""
64-bit unsigned device address, meant to be used as buffer addresses via the
Vulkan feature `buffer_device_address`.
"""
primitive type DeviceAddress 64 end

DeviceAddress(address::UInt64) = reinterpret(DeviceAddress, address)
DeviceAddress(address::Integer) = DeviceAddress(UInt64(address))
Base.UInt64(addr::DeviceAddress) = reinterpret(UInt64, addr)

Base.convert(::Type{UInt64}, address::DeviceAddress) = reinterpret(UInt64, address)
Base.convert(::Type{DeviceAddress}, address::UInt64) = reinterpret(DeviceAddress, address)
Base.convert(::Type{DeviceAddress}, address::Integer) = reinterpret(DeviceAddress, UInt64(address))
Base.:(+)(x::DeviceAddress, y::DeviceAddress) = UInt64(x) + UInt64(y)
Base.:(+)(x::Integer, y::DeviceAddress) = x + UInt64(y)
Base.:(+)(x::DeviceAddress, y::Integer) = UInt64(x) + y
DeviceAddress(ptr::Ptr) = DeviceAddress(UInt64(ptr))

SPIRV.primitive_type_to_spirv(::Type{DeviceAddress}) = UInt64
SPIRV.Pointer{T}(address::DeviceAddress) where {T} = Pointer{T}(convert(UInt64, address))

struct Buffer <: LavaAbstraction
  handle::Vk.Buffer
  size::Int64
  offset::Int64
  stride::Int64
  usage_flags::Vk.BufferUsageFlag
  queue_family_indices::Vector{Int8}
  sharing_mode::Vk.SharingMode
  memory::RefValue{Memory}
  layout::LayoutStrategy
end

vk_handle_type(::Type{Buffer}) = Vk.Buffer

Vk.bind_buffer_memory(buffer::Buffer, memory::Memory) = Vk.bind_buffer_memory(device(buffer), buffer, memory, memory.offset)

isallocated(buffer::Buffer) = isdefined(buffer.memory, 1)

DeviceAddress(buffer::Buffer) = DeviceAddress(Vk.get_buffer_device_address(device(buffer), Vk.BufferDeviceAddressInfo(handle(buffer))) + UInt64(buffer.offset))

function Buffer(
  device,
  size::Integer;
  usage_flags = Vk.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
  queue_family_indices = queue_family_indices(device),
  sharing_mode = Vk.SHARING_MODE_EXCLUSIVE,
  layout::LayoutStrategy = NativeLayout(),
)
  info = Vk.BufferCreateInfo(size, usage_flags, sharing_mode, queue_family_indices)
  handle = unwrap(create(Buffer, device, info))
  Buffer(handle, size, 0, 0, usage_flags, queue_family_indices, sharing_mode, Ref{Memory}(), layout)
end

function Base.show(io::IO, buffer::Buffer)
  print(io, Buffer, "($(Base.format_bytes(buffer.size))), $(buffer.usage_flags)")
  if !isallocated(buffer)
    print(io, ", unallocated")
  end
  print(io, ')')
end

function Base.similar(buffer::Buffer; memory_domain = nothing, usage_flags = buffer.usage_flags, layout = buffer.layout)
  similar = Buffer(device(buffer), buffer.size; usage_flags, buffer.queue_family_indices, buffer.sharing_mode, layout)
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
