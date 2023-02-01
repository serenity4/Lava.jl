"""
Cycle-independent specification of a program invocation for graphics operations.
"""
struct ProgramInvocation
  program::Program
  command::Command
  data::ProgramInvocationData
  targets::Optional{RenderTargets}
  draw_state::Optional{DrawState}
  resource_dependencies::Dictionary{Resource, ResourceDependency}
end

ProgramInvocation(program::Program, command::DispatchCommand, data::ProgramInvocationData, resource_dependencies::Dictionary = Dictionary{Resource, ResourceDependency}()) =
  ProgramInvocation(program, Command(command), data, nothing, nothing, resource_dependencies)
ProgramInvocation(program::Program, command::DrawCommand, data::ProgramInvocationData, targets::RenderTargets, render_state::RenderState, invocation_state::ProgramInvocationState, resource_dependencies::Dictionary = Dictionary{Resource, ResourceDependency}()) =
  ProgramInvocation(program, Command(command), data, targets, DrawState(render_state, invocation_state), resource_dependencies)

function command_info!(allocator::LinearAllocator, device::Device, invocation::ProgramInvocation, node_id::NodeID, materialized_resources)
  data = device_address_block!(allocator, device.descriptors, materialized_resources, node_id, invocation.data)
  CommandInfo(invocation.command, invocation.program, data, invocation.targets, invocation.draw_state)
end

"""
Allocate the provided bytes respecting the specified alignment.

The data must have been properly serialized before hand with the corresponding layout for it to be valid for use inside shaders.
"""
function allocate_data!(allocator::LinearAllocator, bytes::AbstractVector{UInt8}, load_alignment::Integer)
  sub = copyto!(allocator, bytes, load_alignment)
  DeviceAddress(sub)
end

function allocate_data!(allocator::LinearAllocator, bytes::AbstractVector{UInt8}, T::DataType, layout::VulkanLayout)
  # TODO: Check that the SPIR-V type of the load instruction corresponds to the type of `data`.
  # TODO: Get alignment from the extra operand MemoryAccessAligned of the corresponding OpLoad instruction.
  allocate_data!(allocator, bytes, alignment(layout, T))
end

allocate_data!(allocator::LinearAllocator, data::T, layout::VulkanLayout) where {T} = allocate_data!(allocator, serialize(data, layout), T, layout)
allocate_data(allocator::LinearAllocator, program::Program, data::T) where {T} = allocate_data!(allocator, data, program.layout)

"""
Program to be compiled into a pipeline with a specific state.
"""
@auto_hash_equals struct ProgramInstance
  program::Program
  draw_state::Optional{DrawState}
  targets::Optional{RenderTargets}
end
