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
  transfers::Vector{Command}
  presentations::Vector{Command}
end

function record!(record::CompactRecord, command::Command)
  if is_graphics(command)
    (; draw, program, data_address, targets, state) = command.graphics
    program_draws = get!(Dictionary, record.draws, program)
    draws = get!(Vector{Pair{Command,RenderTargets}}, program_draws, (data_address, state))
    push!(draws, command => targets)
  elseif is_compute(command)
    (; dispatch, program, data_address) = command.compute
    program_dispatches = get!(Dictionary, record.dispatches, program)
    dispatches = get!(Vector{Command}, program_dispatches, data_address)
    push!(dispatches, command)
  elseif is_transfer(command)
    push!(record.transfers, command)
  elseif is_present(command)
    push!(record.presentations, command)
  end
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

function initialize_index_buffer(cb::CommandBuffer, device::Device, id::IndexData)
  allocate_index_buffer(id, device)
  Vk.cmd_bind_index_buffer(cb, id.index_buffer[], 0, Vk.INDEX_TYPE_UINT32)
end

function allocate_index_buffer(id::IndexData, device::Device)
  #TODO: Create index buffer in render graph to avoid excessive synchronization.
  id.index_buffer[] = Buffer(device; data = id.index_list, usage_flags = Vk.BUFFER_USAGE_INDEX_BUFFER_BIT)
end
