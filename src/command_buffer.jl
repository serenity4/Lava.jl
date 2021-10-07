abstract type CommandBuffer <: LavaAbstraction end

vk_handle_type(::Type{<:CommandBuffer}) = Vk.CommandBuffer

struct SimpleCommandBuffer <: CommandBuffer
    handle::Vk.CommandBuffer
    queue_family_index::Int
end

function Vk.CommandBufferSubmitInfoKHR(cb::CommandBuffer)
    Vk.CommandBufferSubmitInfoKHR(C_NULL, cb, 1)
end

start_recording(cb::SimpleCommandBuffer) = unwrap(Vk.begin_command_buffer(cb, Vk.CommandBufferBeginInfo(flags = Vk.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)))
end_recording(cb) = unwrap(Vk.end_command_buffer(cb))

struct CommandPools
    device::Vk.Device
    available::Dictionary{Int,Vk.CommandPool}
end

CommandPools(device) = CommandPools(device, Dictionary())

function request_pool!(pools::CommandPools, queue_family_index)
    haskey(pools.available, queue_family_index) && return pools.available[queue_family_index]
    pool = Vk.CommandPool(pools.device, queue_family_index)
    insert!(pools.available, queue_family_index, pool)
    pool
end
