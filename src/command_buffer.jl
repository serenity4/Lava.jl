"""
    @record command_buffer [create_info] commands

Convenience macro for recording a sequence of API commands into a command buffer `cbuff`.
All calls to API commands have `cbuff` inserted as their first argument, and are wrapped inside
`begin_command_buffer(cbuff, info)` and `end_command_buffer(cbuff)`.

!!! warning
    An expression is assumed to be an API command if it begins with `cmd_`.
    Make sure that all functions that you call satisfy this assumption.

The two-argument version of this macro simply passes in a default `CommandBufferBeginInfo()`.
"""
macro record(cbuff, info, cmds)
    api_calls = postwalk(cmds) do ex
        @match ex begin
            :($f($(args...))) => startswith(string(f), "cmd_") ? :($f($cbuff, $(args...))) : ex
            :($f($(args...); $(kwargs...))) => startswith(string(f), "cmd_") ? :($f($cbuff, $(args...); $(kwargs...))) : ex
            _ => ex
        end
    end
    quote
        $(esc(:(Vk.begin_command_buffer($(esc(cbuff)), $(esc(info))))))
        $(esc(api_calls))
        $(esc(:(Vk.end_command_buffer($(esc(cbuff))))))
    end
end

macro record(cbuff, cmds)
    :(@record $cbuff Vk.CommandBufferBeginInfo() $(esc(cmds)))
end

struct CommandBuffer <: LavaAbstraction
    handle::Vk.CommandBuffer
    queue_family_index::Int
end

vk_handle_type(::Type{CommandBuffer}) = Vk.CommandBuffer

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
