module Lava

using Vulkan: Vk, VkCore

using Reexport
using Dictionaries
using Accessors
using MLStyle
using Graphs
using SPIRV
using AutoHashEquals
using LRUCache: LRU
@reexport using ColorTypes: RGB, BGR, RGBA, ARGB, BGRA, ABGR
using FixedPointNumbers

using glslang_jll: glslang_jll
const glslangValidator = glslang_jll.glslangValidator_path

using TimerOutputs
const to = TimerOutput()

@reexport using ResultTypes
@reexport using ResultTypes: iserror
@reexport using SPIRV: Vec, Mat, Arr, Pointer, F, U

const Optional{T} = Union{T,Nothing}

using UUIDs: UUID, uuid1
using Base: RefValue

"""
Abstraction defined in the scope of this package.
"""
abstract type LavaAbstraction end

include("utils.jl")
include("formats.jl")
include("handles.jl")
include("queue_dispatch.jl")
include("synchronization.jl")
include("hashtable.jl")

const debug_callback_c = Ref{Ptr{Cvoid}}(C_NULL)

include("command_buffer.jl")
include("init.jl")
include("memory.jl")
include("buffer.jl")
include("image.jl")
include("allocators.jl")
include("dimensions.jl")
include("attachments.jl")

include("spirv.jl")
include("shaders/dependencies.jl")
include("shaders/formats.jl")
include("shaders/source.jl")
include("shaders/vulkan.jl")
include("shaders/macros.jl")

include("resources.jl")
include("textures.jl")
include("descriptors.jl")
include("pipeline.jl")
include("render_state.jl")
include("device.jl")
include("program.jl")
include("resources/creation.jl")
include("binding_state.jl")
include("wsi.jl")
include("frame.jl")
include("draw.jl")
include("render_graph.jl")
include("bake.jl")
include("command.jl")
include("procedural_api.jl")
include("resources/resolution.jl")
# include("transition.jl")
include("debug.jl")
include("precompile.jl")

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
  Instance, QueueDispatch, set_presentation_queue, Device, init,

  # synchronization
  ExecutionState,
  SubmissionInfo, sync_submission,

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
  Texture, Sampling, DEFAULT_SAMPLING,

  # attachments
  Attachment, READ, WRITE, RenderTargets,

  # attachment dimensions
  SizeUnit, SIZE_ABSOLUTE, SIZE_SWAPCHAIN_RELATIVE, SIZE_VIEWPORT_RELATIVE,

  # program
  Program, ProgramInvocationState,

  # shaders
  ShaderSource, @shader,
  @fragment, @vertex,
  ShaderCache,
  Shader,

  # resources
  new!,
  buffer, image, attachment,
  LogicalBuffer, LogicalImage, LogicalAttachment,
  PhysicalBuffer, PhysicalImage, PhysicalAttachment,

  # descriptors
  PhysicalDescriptors, ResourceMetaConfig,
  index,

  # render state
  RenderState,

  # commands
  CompactRecord, draw,
  DrawCommand, DrawInfo, DrawState, DrawDirect, DrawIndexed, DrawIndirect, DrawIndexedIndirect,
  allocate_data, DrawData, request_descriptor_index,
  DeviceAddress,
  StatefulRecording, set_program, invocation_state, set_invocation_state, render_state, set_render_state, set_data,

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

  # WSI
  Surface, Swapchain,

  # frame
  Frame, FrameCycle, cycle!, acquire_next_image,

  # SPIR-V reexports
  ShaderInterface, Decorations, Vec, Arr, Mat, Vec2, Vec3, Vec4, Mat2, Mat3, Mat4
end
