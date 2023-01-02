abstract type DispatchCommand <: CommandImplementation end

struct Dispatch <: DispatchCommand
  group_sizes::NTuple{3,Int}
end
Dispatch(x::Integer, y::Integer = 1, z::Integer = 1) = Dispatch((x, y, z))

apply(cb::CommandBuffer, dispatch::Dispatch) = Vk.cmd_dispatch(cb, dispatch.group_sizes...)

struct DispatchIndirect <: DispatchCommand
  buffer::Buffer
end

apply(cb::CommandBuffer, dispatch::DispatchIndirect) = Vk.cmd_dispatch_indirect(cb, dispatch.buffer, dispatch.buffer.offset)
