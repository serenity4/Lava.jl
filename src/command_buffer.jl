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
        $(esc(:(begin_command_buffer($(esc(cbuff)), $(esc(info))))))
        $(esc(api_calls))
        $(esc(:(end_command_buffer($(esc(cbuff))))))
    end
end

macro record(cbuff, cmds)
    :(@record $cbuff CommandBufferBeginInfo() $(esc(cmds)))
end

"""
Manages one command pool per thread.
"""
struct ThreadedCommandPool
    pools::Dictionary{Int,Vk.CommandPool}
end

function ThreadedCommandPool(device, disp::QueueDispatch, usage = dictionary(1:Base.Threads.nthreads() .=> QUEUE_COMPUTE_BIT | QUEUE_GRAPHICS_BIT | QUEUE_TRANSFER_BIT))
    pools = dictionary(thread => Vk.CommandPool(device, info(queue(disp, usage[thread])).queue_family_index) for thread in keys(usage))
    ThreadedCommandPool(pools)
end

function Vk.CommandPool(pool::ThreadedCommandPool)
    pool.pools[Base.Threads.threadid()]
end

Base.convert(T::Type{Vk.CommandPool}, pool::ThreadedCommandPool) = T(pool)
