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

function command_info!(allocator::LinearAllocator, gdescs::GlobalDescriptors, invocation::ProgramInvocation, node_id::NodeID, device::Device)
  data = device_address_block!(allocator, gdescs, node_id, invocation.data, invocation.program.type_info, device.layout)
  CommandInfo(invocation.command, invocation.program, data, invocation.targets, invocation.draw_state)
end

"""
Allocate the provided bytes respecting the specified alignment.

The data must have been properly padded before hand with the correct shader offsets for it to be usable inside shaders.
"""
function allocate_data!(allocator::LinearAllocator, bytes::AbstractVector{UInt8}, load_alignment::Integer)
  sub = copyto!(allocator, bytes, load_alignment)
  DeviceAddress(sub)
end

"""
Allocate the provided bytes with an alignment computed from `layout`.

Padding will be applied to composite and array types using the offsets specified in the shader.
"""
function allocate_data!(allocator::LinearAllocator, type_info::TypeInfo, bytes::AbstractVector{UInt8}, t::SPIRType, layout::VulkanLayout, align_bytes = isa(t, StructType) || isa(t, ArrayType))
  align_bytes && (bytes = align(bytes, t, type_info))
  # TODO: Check that the SPIR-V type of the load instruction corresponds to the type of `data`.
  # TODO: Get alignment from the extra operand MemoryAccessAligned of the corresponding OpLoad instruction.
  load_alignment = data_alignment(layout, t)
  allocate_data!(allocator, bytes, load_alignment)
end

data_alignment(layout::VulkanLayout, t::SPIRType) = alignment(layout, t, [SPIRV.StorageClassPhysicalStorageBuffer], false)

allocate_data!(allocator::LinearAllocator, data::T, type_info::TypeInfo, layout::VulkanLayout) where {T} = allocate_data!(allocator, type_info, extract_bytes(data), type_info.tmap[T], layout)

allocate_data(allocator::LinearAllocator, program::Program, data::T, layout::VulkanLayout) where {T} =
  allocate_data!(allocator, data, program.type_info, layout)

"""
Program to be compiled into a pipeline with a specific state.
"""
@auto_hash_equals struct ProgramInstance
  program::Program
  draw_state::Optional{DrawState}
  targets::Optional{RenderTargets}
end
