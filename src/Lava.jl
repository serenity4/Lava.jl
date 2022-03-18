module Lava

using Vulkan: Vk, VkCore

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

using UUIDs: UUID, uuid1

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
include("descriptors.jl")

include("spirv.jl")
include("shaders/dependencies.jl")
include("shaders/formats.jl")
include("shaders/source.jl")
include("shaders/vulkan.jl")

include("resources.jl")
include("pipeline.jl")
include("render_state.jl")
include("program.jl")
include("device.jl")
include("resources/creation.jl")
include("binding_state.jl")
include("frame.jl")
include("render_graph.jl")
include("bake.jl")
include("resources/resolution.jl")
# include("transition.jl")
include("command.jl")
include("debug.jl")

function __init__()
  # for debugging in Vulkan
  debug_callback_c[] =
    @cfunction(
      debug_callback,
      UInt32,
      (Vk.DebugUtilsMessageSeverityFlagEXT, Vk.DebugUtilsMessageTypeFlagEXT, Ptr{VkCore.VkDebugUtilsMessengerCallbackDataEXT}, Ptr{Cvoid})
    )
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

  # allocators
  LinearAllocator, available_size,

  # buffers
  Buffer, DenseBuffer, BufferBlock, SubBuffer,
  device_address, allocate!, isallocated, bind!,
  transfer,

  # images
  Image, ImageBlock, View, ImageView,

  # textures
  Texture, DEFAULT_SAMPLING,

  # attachments
  Attachment, READ, WRITE, RenderTargets,

  # attachment dimensions
  SizeUnit, SIZE_ABSOLUTE, SIZE_SWAPCHAIN_RELATIVE, SIZE_VIEWPORT_RELATIVE,

  # program
  Program, ProgramInvocationState,

  # shaders
  ShaderSource, @shader,
  ShaderCache,
  Shader,

  # resources
  new!,
  buffer, image, attachment,
  LogicalBuffer, LogicalImage, LogicalAttachment,
  PhysicalBuffer, PhysicalImage, PhysicalAttachment,

  # descriptors
  ResourceDescriptors, ResourceMetaConfig,

  # render state
  RenderState,

  # commands
  CompactRecord, draw,
  DrawCommand, Draw, DrawIndexed, DrawIndirect, DrawIndexedIndirect,
  set_program, draw_state, set_draw_state, set_material,
  DrawData,

  # render graph
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
  RenderGraph, render,
  RenderArea, RenderNode,
  ResourceDependency, add_resource_dependency, add_resource_dependencies, @add_resource_dependencies, clear_attachments,

  # SPIR-V reexports
  ShaderInterface
end
