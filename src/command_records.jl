"""
Type that records command lazily, for them to be flushed into an Vulkan command buffer later.
"""
abstract type CommandRecord <: LavaAbstraction end

"""
Record that compacts action commands according to their program and state before flushing.

This allows to group draw calls that use the exact same rendering state for better performance.
"""
struct CompactRecord <: CommandRecord
  node::RenderNode
  draws::Dictionary{Program,Dictionary{Tuple{DeviceAddressBlock, DrawState},Vector{Pair{Command,RenderTargets}}}}
  dispatches::Dictionary{Program,Dictionary{DeviceAddressBlock, Vector{Command}}}
end

draw!(record::CommandRecord, info::CommandInfo) = draw!(record, info.command, info.program, info.data, info.targets, info.state)
function draw!(record::CompactRecord, command::Command, program::Program, data::DeviceAddressBlock, targets::RenderTargets, state::DrawState)
  program_draws = get!(Dictionary, record.draws, program)
  draws = get!(Vector{Pair{Command,RenderTargets}}, program_draws, (data, state))
  push!(draws, command => targets)
  nothing
end
dispatch!(record::CommandRecord, info::CommandInfo) = dispatch!(record, info.command, info.program, info.data)
function dispatch!(record::CompactRecord, command::Command, program::Program, data::DeviceAddressBlock)
  program_dispatches = get!(Dictionary, record.dispatches, program)
  dispatches = get!(Vector{Command}, program_dispatches, data)
  push!(dispatches, command)
  nothing
end

Base.show(io::IO, record::CompactRecord) = print(
  io,
  CompactRecord,
  '(',
  length(record.draws) + length(record.dispatches),
  " programs",
  ", ", sum(x -> sum(length, values(x); init = 0), values(record.draws); init = 0), " draw commands",
  ", ", sum(x -> sum(length, values(x); init = 0), values(record.dispatches); init = 0), " compute dispatches",
  ')',
)

function draw_command(program::Program, data_address, idata, color...; depth = nothing, stencil = nothing, instances = 1:1, render_state::RenderState = RenderState(), invocation_state::ProgramInvocationState = ProgramInvocationState())
  state = DrawState(render_state, invocation_state)
  command = Command(DrawIndexed(idata; instances))
  targets = RenderTargets(color...; depth, stencil)
  CommandInfo(command, program, data_address, targets, state)
end

function initialize(cb::CommandBuffer, device::Device, id::IndexData)
  allocate_index_buffer(id, device)
  Vk.cmd_bind_index_buffer(cb, id.index_buffer[], 0, Vk.INDEX_TYPE_UINT32)
end

function allocate_index_buffer(id::IndexData, device::Device)
  #TODO: Create index buffer in render graph to avoid excessive synchronization.
  id.index_buffer[] = Buffer(device; data = id.index_list, usage_flags = Vk.BUFFER_USAGE_INDEX_BUFFER_BIT)
end
