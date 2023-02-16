abstract type DispatchCommand end

mutable struct ComputeCommand <: CommandImplementation
  const dispatch::DispatchCommand
  const program::Program
  const data::Optional{ProgramInvocationData}
  data_address::DeviceAddressBlock
  const resource_dependencies::Dictionary{Resource, ResourceDependency}
end

resource_dependencies(command::ComputeCommand) = command.resource_dependencies

ComputeCommand(dispatch::DispatchCommand, program::Program, data::ProgramInvocationData, resource_dependencies = Dictionary{Resource, ResourceDependency}()) =
  ComputeCommand(dispatch, program, data, DeviceAddressBlock(0), resource_dependencies)

ComputeCommand(dispatch::DispatchCommand, program::Program, data_address::DeviceAddressBlock, resource_dependencies = Dictionary{Resource, ResourceDependency}()) =
  ComputeCommand(dispatch, program, nothing, data_address, resource_dependencies)

struct Dispatch <: DispatchCommand
  group_sizes::NTuple{3,Int}
end
Dispatch(x::Integer, y::Integer = 1, z::Integer = 1) = Dispatch((x, y, z))

apply(cb::CommandBuffer, dispatch::Dispatch) = Vk.cmd_dispatch(cb, dispatch.group_sizes...)

struct DispatchIndirect <: DispatchCommand
  buffer::Resource
end

function apply(cb::CommandBuffer, dispatch::DispatchIndirect, resources)
  (; buffer) = get_physical_resource(resources, dispatch.buffer)
  Vk.cmd_dispatch_indirect(cb, buffer, buffer.offset)
end
