module Lava

using Core: MethodInstance
using BitMasks
using Vulkan: Vk, VkCore, format_type
using StyledStrings

using ForwardMethods
using CompileTraces
using SnoopPrecompile
using Reexport
using Dictionaries
using Accessors
using MLStyle
using Graphs
using SPIRV
using StructEquality
using LRUCache: LRU
@reexport using ColorTypes: RGB, BGR, RGBA, ARGB, BGRA, ABGR
using FixedPointNumbers

using glslang_jll: glslang_jll
const glslangValidator = glslang_jll.glslangValidator_path

using TimerOutputs
const to = TimerOutput()

@reexport using ResultTypes
@reexport using ResultTypes: iserror
@reexport using SPIRV: Vec, Mat, Arr, Pointer, F, U, @load, @store, ShaderExecutionOptions, FragmentExecutionOptions, ComputeExecutionOptions, GeometryExecutionOptions, TessellationExecutionOptions, MeshExecutionOptions, CommonExecutionOptions, VulkanLayout, VulkanAlignment

const Optional{T} = Union{T,Nothing}

using UUIDs: UUID, uuid1
using Base: RefValue

"""
Abstraction defined in the scope of this package.
"""
abstract type LavaAbstraction end

include("utils.jl")
include("handles.jl")
include("queue_dispatch.jl")
include("synchronization.jl")
include("hashtable.jl")
include("cache.jl")

const debug_callback_c = Ref{Ptr{Cvoid}}(C_NULL)

include("command_buffer.jl")
include("init.jl")
include("memory.jl")
include("buffer.jl")
include("subresource.jl")
include("image.jl")
include("allocators.jl")
include("dimensions.jl")
include("attachments.jl")

include("shader.jl")

include("program.jl")
include("resources.jl")
include("textures.jl")
include("descriptors.jl")
include("render_state.jl")
include("invocation_data.jl")
include("command.jl")
include("pipeline.jl")
include("device.jl")
include("resources/creation.jl")
include("bind_state.jl")
include("wsi.jl")
include("frame.jl")
include("render_graph.jl")
include("command_records.jl")
include("bake.jl")
include("node_synchronization.jl")
include("flush.jl")
include("resources/resolution.jl")
include("debug.jl")

@precompile_all_calls @compile_traces verbose = false joinpath(pkgdir(Lava), "traces", "precompilation_traces.jl")

function __init__()
  # for debugging in Vulkan
  debug_callback_c[] =
    @cfunction(
      debug_callback,
      UInt32,
      (Vk.DebugUtilsMessageSeverityFlagEXT, Vk.DebugUtilsMessageTypeFlagEXT, Ptr{VkCore.VkDebugUtilsMessengerCallbackDataEXT}, Ptr{Cvoid})
    )
end

export
  Vk,
  Instance, QueueDispatch, set_presentation_queue, Device, init,

  # synchronization
  ExecutionState,
  SubmissionInfo, sync_submission,

  # memory
  Memory,
  MemoryDomain, MEMORY_DOMAIN_DEVICE, MEMORY_DOMAIN_HOST, MEMORY_DOMAIN_HOST_CACHED,
  ismapped,

  # allocators
  LinearAllocator, available_size,

  # buffers
  LogicalBuffer, Buffer,
  device_address, allocate!, isallocated, bind!,
  transfer,

  # subresources
  Subresource, SubresourceMap,
  LayerRange, layer_range,
  MipRange, mip_range,

  # images
  LogicalImage, Image, ImageView,

  # textures
  Texture, Sampling, DEFAULT_SAMPLING,

  # attachments
  LogicalAttachment, Attachment, READ, WRITE, RenderTargets,

  # attachment dimensions
  SizeUnit, SIZE_ABSOLUTE, SIZE_SWAPCHAIN_RELATIVE, SIZE_VIEWPORT_RELATIVE, dimensions,

  # program
  Program, ProgramInvocationState, ProgramInvocationData, DataBlock, InvocationDataContext, @invocation_data,

  # shaders
  ShaderSource, @shader,
  # @fragment, @vertex, @compute, etc just like SPIR-V but higher-level - see shader.jl.
  ShaderCache,
  Shader,

  # resources
  Resource, buffer_resource, image_resource, attachment_resource, assert_type,
  RESOURCE_TYPE_BUFFER, RESOURCE_TYPE_IMAGE, RESOURCE_TYPE_ATTACHMENT,

  # descriptors
  Descriptor, DescriptorID,
  storage_image_descriptor, sampler_descriptor, sampled_image_descriptor, texture_descriptor,
  GlobalDescriptors, GlobalDescriptorsConfig,
  GLOBAL_DESCRIPTOR_SET_INDEX,
  BINDING_SAMPLED_IMAGE, BINDING_STORAGE_IMAGE, BINDING_SAMPLER_IMAGE, BINDING_COMBINED_IMAGE_SAMPLER,

  # render state
  RenderState,

  # commands
  CompactRecord, Command, record!,
  graphics_command, draw!, GraphicsCommand, DrawState, DrawIndexed, DrawIndirect, DrawIndexedIndirect,
  compute_command, dispatch!, ComputeCommand, Dispatch, DispatchIndirect,
  transfer_command, transfer!, TransferCommand,
  present_command, present!, PresentCommand,
  DescriptorIndex,
  DeviceAddressBlock, DeviceAddress,
  StatefulRecording, set_program, invocation_state, set_invocation_state, render_state, set_render_state, set_data,

  # render graph
  ResourceUsageType,
  RESOURCE_USAGE_VERTEX_BUFFER,
  RESOURCE_USAGE_INDEX_BUFFER,
  RESOURCE_USAGE_COLOR_ATTACHMENT,
  RESOURCE_USAGE_DEPTH_ATTACHMENT,
  RESOURCE_USAGE_STENCIL_ATTACHMENT,
  RESOURCE_USAGE_INPUT_ATTACHMENT,
  RESOURCE_USAGE_TEXTURE,
  RESOURCE_USAGE_BUFFER,
  RESOURCE_USAGE_PHYSICAL_BUFFER,
  RESOURCE_USAGE_IMAGE,
  RESOURCE_USAGE_DYNAMIC,
  RESOURCE_USAGE_STORAGE,
  RESOURCE_USAGE_TEXEL,
  RESOURCE_USAGE_UNIFORM,
  RESOURCE_USAGE_SAMPLER,
  RenderGraph, render!, render,
  RenderArea, RenderNode, add_node!, add_nodes!,
  ResourceDependency, add_resource_dependency!, add_resource_dependencies!, @add_resource_dependencies, @resource_dependencies, clear_attachments, ClearValue, DEFAULT_CLEAR_VALUE,

  # WSI
  Surface, Swapchain,

  # frame
  Frame, FrameCycle, cycle!, recreate!, acquire_next_image, draw_and_prepare_for_presentation,

  # SPIR-V reexports
  ShaderInterface, Decorations, Vec, Arr, Mat, Vec2, Vec3, Vec4, Mat2, Mat3, Mat4
end
