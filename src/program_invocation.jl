struct ResourceDependency
  type::ShaderResourceType
  access::MemoryAccess
  clear_value::Optional{NTuple{4,Float32}}
  samples::Int64
end
ResourceDependency(type, access; clear_value = nothing, samples = 1) = ResourceDependency(type, access, clear_value, samples)

function Base.merge(x::ResourceDependency, y::ResourceDependency)
  @assert x.id === y.id
  ResourceDependency(x.id, x.type | y.type, x.access | y.access)
end

"""
Cycle-independent specification of a program invocation for graphics operations.
"""
struct ProgramInvocation
  program::Program
  command::Command
  data::ProgramInvocationData
  targets::Optional{RenderTargets}
  state::Optional{DrawState}
  resource_dependencies::Dictionary{Resource, ResourceDependency}
end

ProgramInvocation(program::Program, command::DispatchCommand, data::ProgramInvocationData, resource_dependencies::Dictionary = Dictionary{Resource, ResourceDependency}()) =
  ProgramInvocation(program, Command(command), data, nothing, nothing, resource_dependencies)
ProgramInvocation(program::Program, command::DrawCommand, data::ProgramInvocationData, targets::RenderTargets, render_state::RenderState, invocation_state::ProgramInvocationState, resource_dependencies::Dictionary = Dictionary{Resource, ResourceDependency}()) =
  ProgramInvocation(program, Command(command), data, targets, DrawState(render_state, invocation_state), resource_dependencies)

function command_info!(allocator::LinearAllocator, device::Device, invocation::ProgramInvocation, node_id::NodeID, materialized_resources)
  data = device_address_block!(allocator, device.descriptors, materialized_resources, node_id, invocation.data)
  CommandInfo(invocation.command, invocation.program, data, invocation.targets, invocation.state)
end

"""
Program to be compiled into a pipeline with a specific state.
"""
@auto_hash_equals struct ProgramInstance
  program::Program
  state::Optional{DrawState}
  targets::Optional{RenderTargets}
end
