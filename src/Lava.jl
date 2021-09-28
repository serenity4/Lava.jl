module Lava

import Vulkan as Vk

using Reexport
using Dictionaries
using Accessors
using MLStyle
using LightGraphs, MetaGraphs
using XCB
using SPIRV
using glslang_jll: glslang_jll

const glslangValidator = glslang_jll.glslangValidator_path

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
include("hashtable.jl")

const debug_callback_c = Ref{Ptr{Cvoid}}(C_NULL)

function __init__()
    # for debugging in Vulkan
    debug_callback_c[] =
        @cfunction(Vk.default_debug_callback, UInt32, (Vk.DebugUtilsMessageSeverityFlagEXT, Vk.DebugUtilsMessageTypeFlagEXT, Ptr{Vk.core.VkDebugUtilsMessengerCallbackDataEXT}, Ptr{Cvoid}))
end

include("command_buffer.jl")
include("init.jl")
include("memory.jl")
include("buffer.jl")
include("image.jl")
include("allocators.jl")
include("dimensions.jl")
include("attachments.jl")
include("textures.jl")
include("render_pass.jl")
include("descriptors.jl")
include("pipeline.jl")

include("shaders/dependencies.jl")
# include("shaders/resources.jl")
include("shaders/formats.jl")
include("shaders/specification.jl")
include("shaders/source.jl")
include("shaders/compilation.jl")

include("device.jl")
include("program.jl")
include("render_state.jl")
include("binding_state.jl")
include("frame_graph.jl")
include("resources.jl")
include("command.jl")
# include("frames.jl")
include("wsi.jl")

# include("synchronization.jl")

# include("vulkan.jl")

export
        Vk,
        Instance, QueueDispatch, Device, init,

        # memory
        Memory, DenseMemory, MemoryBlock, SubMemory,
        memory,
        MemoryDomain, MEMORY_DOMAIN_DEVICE, MEMORY_DOMAIN_HOST, MEMORY_DOMAIN_HOST_CACHED,

        # buffers
        Buffer, DenseBuffer, BufferBlock, SubBuffer,
        device_address, allocate!, isallocated, bind!,

        # images
        Image, ImageBlock, View, ImageView,

        # textures
        Texture, DEFAULT_SAMPLING,

        # attachment dimensions
        SizeUnit, SIZE_ABSOLUTE, SIZE_SWAPCHAIN_RELATIVE, SIZE_VIEWPORT_RELATIVE,

        # program
        Program, ProgramInterface, ProgramInvocationState,

        # shaders
        ShaderDependencies,
        ShaderResource,
        ShaderLanguage, SPIR_V, GLSL, HLSL,
        ShaderSpecification,
        ShaderCache,
        Shader,

        # descriptors
        ResourceDescriptors, ResourceMetaConfig,

        # render state
        RenderState,

        # commands
        CompactRecord, draw,
        DrawCommand, Draw, DrawIndexed, DrawIndirect, DrawIndexedIndirect,
        set_program, set_state,

        # frame graph
        BufferResourceInfo, ImageResourceInfo, AttachmentResourceInfo,
        ResourceInfo, add_resource!,
        ResourceType,
        RESOURCE_TYPE_VERTEX_BUFFER,
        RESOURCE_TYPE_INDEX_BUFFER,
        RESOURCE_TYPE_COLOR_ATTACHMENT,
        RESOURCE_TYPE_DEPTH_ATTACHMENT,
        RESOURCE_TYPE_STENCIL_ATTACHMENT,
        RESOURCE_TYPE_INPUT_ATTACHMENT,
        RESOURCE_TYPE_TEXTURE,
        RESOURCE_TYPE_BUFFER,
        RESOURCE_TYPE_IMAGE,
        RESOURCE_TYPE_DYNAMIC,
        RESOURCE_TYPE_STORAGE,
        RESOURCE_TYPE_TEXEL,
        RESOURCE_TYPE_UNIFORM,
        RESOURCE_TYPE_SAMPLER,
        Pass, add_pass!,
        ResourceUsage, add_resource_usage!, resource_usages, @resource_usages,
        FrameGraph, render
end
