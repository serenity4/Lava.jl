module Lava

import Vulkan as Vk

using Reexport
using Dictionaries
using Accessors
using MLStyle
using Graphs, MetaGraphs
using XCB
using Transducers
using SPIRV
using AutoHashEquals

using glslang_jll: glslang_jll
const glslangValidator = glslang_jll.glslangValidator_path

using TimerOutputs
const to = TimerOutput()

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
include("synchronization.jl")
include("hashtable.jl")

const debug_callback_c = Ref{Ptr{Cvoid}}(C_NULL)

include("command_buffer.jl")
include("init.jl")
include("wsi.jl")
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
include("shaders/formats.jl")
include("shaders/source.jl")
include("shaders/vulkan.jl")

include("device.jl")
include("render_state.jl")
include("program.jl")
include("binding_state.jl")
include("frame.jl")
include("frame_graph.jl")
include("resource_resolution.jl")
include("command.jl")
include("debug.jl")

function __init__()
# for debugging in Vulkan
    debug_callback_c[] =
        @cfunction(debug_callback, UInt32, (Vk.DebugUtilsMessageSeverityFlagEXT, Vk.DebugUtilsMessageTypeFlagEXT, Ptr{Vk.core.VkDebugUtilsMessengerCallbackDataEXT}, Ptr{Cvoid}))
end

# include("vulkan.jl")

export
        Vk,
        Instance, QueueDispatch, Device, init,

        # synchronization
        ExecutionState,

        # memory
        Memory, DenseMemory, MemoryBlock, SubMemory,
        memory,
        MemoryDomain, MEMORY_DOMAIN_DEVICE, MEMORY_DOMAIN_HOST, MEMORY_DOMAIN_HOST_CACHED,
        offset, ismapped,

        # buffers
        Buffer, DenseBuffer, BufferBlock, SubBuffer,
        device_address, allocate!, isallocated, bind!,
        buffer, transfer,

        # images
        Image, ImageBlock, View, ImageView, image,

        # textures
        Texture, DEFAULT_SAMPLING,

        # attachments
        Attachment, READ, WRITE, TargetAttachments,

        # attachment dimensions
        SizeUnit, SIZE_ABSOLUTE, SIZE_SWAPCHAIN_RELATIVE, SIZE_VIEWPORT_RELATIVE,

        # program
        Program, ProgramInterface, ProgramInvocationState,

        # shaders
        ShaderSource, @shader,
        ShaderCache,
        Shader,

        # descriptors
        ResourceDescriptors, ResourceMetaConfig,

        # render pass
        RenderPass,

        # render state
        RenderState,

        # commands
        CompactRecord, draw,
        DrawCommand, Draw, DrawIndexed, DrawIndirect, DrawIndexedIndirect,
        set_program, draw_state, set_draw_state, set_material,
        DrawData,

        # frame
        Frame, Resource, register,

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
        ResourceUsage, add_resource_usage!, resource_usages, @resource_usages, clear_attachments,
        FrameGraph, render,

        # SPIR-V reexports
        ShaderInterface
end
