@enum CommandType::UInt8 begin
  COMMAND_TYPE_DRAW_INDEXED = 1
  COMMAND_TYPE_DRAW_INDIRECT = 2
  COMMAND_TYPE_DRAW_INDEXED_INDIRECT = 3
  COMMAND_TYPE_DISPATCH = 10
  COMMAND_TYPE_DISPATCH_INDIRECT = 11
  COMMAND_TYPE_PRESENT = 20
  COMMAND_TYPE_TRANSFER = 30
end

abstract type CommandImplementation end

mutable struct Command
  const type::CommandType
  const impl::CommandImplementation
end

Base.iterate(command::Command) = (command, nothing)
Base.iterate(command::Command, state) = nothing

get_physical_resource(resources, resource::Resource) = islogical(resource) ? resources[resource.id] : resource

struct ResourceDependency
  type::ResourceUsageType
  access::MemoryAccess
  clear_value::Optional{ClearValue}
  samples::Optional{Int64}
end
ResourceDependency(type, access; clear_value = nothing, samples = nothing) = ResourceDependency(type, access, clear_value, samples)

function Base.merge(x::ResourceDependency, y::ResourceDependency)
  @assert x.id === y.id
  ResourceDependency(x.id, x.type | y.type, x.access | y.access, merge_sample_count(x.samples, y.samples))
end

function merge_sample_count(xsamples, ysamples)
  isnothing(xsamples) || isnothing(ysamples) || xsamples == ysamples || error("Cannot merge resource dependencies with inconsistent sampling parameters ($(xsamples) samples ≠ $(ysamples) samples).")
  something(xsamples, ysamples, Some(nothing))
end

include("command/graphics.jl")
include("command/compute.jl")
include("command/transfer.jl")
include("command/present.jl")

function CommandType(graphics::GraphicsCommand)
  isa(graphics.draw, DrawIndexed) && return COMMAND_TYPE_DRAW_INDEXED
  isa(graphics.draw, DrawIndexedIndirect) && return COMMAND_TYPE_DRAW_INDEXED_INDIRECT
  isa(graphics.draw, DrawIndirect) && return COMMAND_TYPE_DRAW_INDIRECT
  @assert false
end
function CommandType(compute::ComputeCommand)
  isa(compute.dispatch, Dispatch) && return COMMAND_TYPE_DISPATCH
  isa(compute.dispatch, DispatchIndirect) && return COMMAND_TYPE_DISPATCH_INDIRECT
  @assert false
end
is_graphics(command::Command) = in(command.type, (COMMAND_TYPE_DRAW_INDEXED, COMMAND_TYPE_DRAW_INDIRECT, COMMAND_TYPE_DRAW_INDEXED_INDIRECT))
is_compute(command::Command) = in(command.type, (COMMAND_TYPE_DISPATCH, COMMAND_TYPE_DISPATCH_INDIRECT))
is_presentation(command::Command) = command.type == COMMAND_TYPE_PRESENT
is_transfer(command::Command) = command.type == COMMAND_TYPE_TRANSFER
function graphics_command(args...; kwargs...)
  command = GraphicsCommand(args...; kwargs...)
  Command(CommandType(command), command)
end
function compute_command(args...; kwargs...)
  command = ComputeCommand(args...; kwargs...)
  Command(CommandType(command), command)
end
transfer_command(args...; kwargs...) = Command(COMMAND_TYPE_TRANSFER, TransferCommand(args...; kwargs...))
present_command(args...; kwargs...) = Command(COMMAND_TYPE_PRESENT, PresentCommand(args...; kwargs...))

@inline function Base.getproperty(command::Command, name::Symbol)
  name === :graphics && return command.impl::GraphicsCommand
  name === :compute && return command.impl::ComputeCommand
  name === :present && return command.impl::PresentCommand
  name === :transfer && return command.impl::TransferCommand
  name === :any && return command.impl::Union{GraphicsCommand, ComputeCommand, PresentCommand, TransferCommand}
  getfield(command, name)
end

function resource_dependencies(command::Command)
  is_graphics(command) && return resource_dependencies(command.graphics)
  is_compute(command) && return resource_dependencies(command.compute)
  is_presentation(command) && return resource_dependencies(command.present)
  is_transfer(command) && return resource_dependencies(command.transfer)
  @assert false
end
