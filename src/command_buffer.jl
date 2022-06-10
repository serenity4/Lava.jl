abstract type CommandBuffer <: LavaAbstraction end

vk_handle_type(::Type{<:CommandBuffer}) = Vk.CommandBuffer

struct SimpleCommandBuffer <: CommandBuffer
  handle::Vk.CommandBuffer
  queue_family_index::Int64
  queues::QueueDispatch
  to_preserve::Vector{Any}
  to_free::Vector{Any}
end

SimpleCommandBuffer(handle, queue_family_index, queues) = SimpleCommandBuffer(handle, queue_family_index, queues, [], [])

function Vk.CommandBufferSubmitInfo(cb::CommandBuffer)
  Vk.CommandBufferSubmitInfo(C_NULL, cb, 1)
end

start_recording(cb::SimpleCommandBuffer) =
  unwrap(Vk.begin_command_buffer(cb, Vk.CommandBufferBeginInfo(flags = Vk.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)))
end_recording(cb) = unwrap(Vk.end_command_buffer(cb))

SubmissionInfo(cb::SimpleCommandBuffer) = SubmissionInfo(command_buffers = [Vk.CommandBufferSubmitInfo(cb)], release_after_completion = cb.to_preserve, free_after_completion = cb.to_free)

function submit(command_buffer::SimpleCommandBuffer, info::SubmissionInfo = SubmissionInfo())
  push!(info.command_buffers, Vk.CommandBufferSubmitInfo(command_buffer))
  append!(info.release_after_completion, command_buffer.to_preserve)
  append!(info.free_after_completion, command_buffer.to_free)
  submit(command_buffer.queues, command_buffer.queue_family_index, info)
end

struct CommandPools
  device::Vk.Device
  available::Dictionary{Int64,Vk.CommandPool}
end

CommandPools(device) = CommandPools(device, Dictionary())

function request_pool!(pools::CommandPools, queue_family_index)
  haskey(pools.available, queue_family_index) && return pools.available[queue_family_index]
  pool = Vk.CommandPool(pools.device, queue_family_index)
  finalizer(x -> Vk.reset_command_pool(x.device, x), pool)
  insert!(pools.available, queue_family_index, pool)
  pool
end
