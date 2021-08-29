module Lava

import Vulkan as Vk

using Reexport
using Dictionaries
using MLStyle
@reexport using ResultTypes
@reexport using ResultTypes: iserror

const Optional{T} = Union{T,Nothing}

"""
Abstraction defined in the scope of this package.
"""
abstract type LavaAbstraction end

include("utils.jl")
include("handles.jl")
include("queue_dispatch.jl")

const debug_callback_c = Ref{Ptr{Cvoid}}(C_NULL)

function __init__()
    # for debugging in Vulkan
    debug_callback_c[] =
        @cfunction(Vk.default_debug_callback, UInt32, (Vk.DebugUtilsMessageSeverityFlagEXT, Vk.DebugUtilsMessageTypeFlagEXT, Ptr{Vk.core.VkDebugUtilsMessengerCallbackDataEXT}, Ptr{Cvoid}))
end

include("init.jl")
include("memory.jl")
include("buffer.jl")
include("image.jl")
# include("command.jl")
# include("pipeline.jl")
# include("pool.jl")
# include("shader.jl")
# include("synchronization.jl")

# include("vulkan.jl")
# include("api.jl")

export
        LavaAbstraction, Vk,
        Instance, QueueDispatch, Device, init,
        Memory, DenseMemory, MemoryBlock, SubMemory,
        memory,
        MemoryDomain, MEMORY_DOMAIN_DEVICE, MEMORY_DOMAIN_HOST, MEMORY_DOMAIN_HOST_CACHED,
        Buffer, DenseBuffer, BufferBlock, SubBuffer,
        allocate!, isallocated,
        Image, ImageBlock, View, ImageView

end
