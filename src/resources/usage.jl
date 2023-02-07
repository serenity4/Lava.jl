@bitmask_flag ShaderResourceType::UInt16 begin
  SHADER_RESOURCE_TYPE_VERTEX_BUFFER = 1
  SHADER_RESOURCE_TYPE_INDEX_BUFFER = 2
  SHADER_RESOURCE_TYPE_COLOR_ATTACHMENT = 4
  SHADER_RESOURCE_TYPE_DEPTH_ATTACHMENT = 8
  SHADER_RESOURCE_TYPE_STENCIL_ATTACHMENT = 16
  SHADER_RESOURCE_TYPE_INPUT_ATTACHMENT = 32
  SHADER_RESOURCE_TYPE_TEXTURE = 64
  SHADER_RESOURCE_TYPE_BUFFER = 128
  SHADER_RESOURCE_TYPE_IMAGE = 256
  SHADER_RESOURCE_TYPE_DYNAMIC = 512
  SHADER_RESOURCE_TYPE_STORAGE = 1024
  SHADER_RESOURCE_TYPE_TEXEL = 2048
  SHADER_RESOURCE_TYPE_UNIFORM = 4096
  SHADER_RESOURCE_TYPE_SAMPLER = 8192
  SHADER_RESOURCE_TYPE_PHYSICAL_BUFFER = 16384
end

Base.@kwdef struct BufferUsage
  type::ShaderResourceType = ShaderResourceType(0)
  access::MemoryAccess = MemoryAccess(0)
  stages::Vk.PipelineStageFlag2 = Vk.PIPELINE_STAGE_2_NONE
  usage_flags::Vk.BufferUsageFlag = Vk.BufferUsageFlag(0)
end

combine(x::BufferUsage, y::BufferUsage) = BufferUsage(x.type | y.type, x.access | y.access, x.stages | y.stages, x.usage_flags | y.usage_flags)

Base.@kwdef struct ImageUsage
  type::ShaderResourceType = ShaderResourceType(0)
  access::MemoryAccess = MemoryAccess(0)
  stages::Vk.PipelineStageFlag2 = Vk.PIPELINE_STAGE_2_NONE
  usage_flags::Vk.ImageUsageFlag = Vk.ImageUsageFlag(0)
  samples::Int64 = 1
  function ImageUsage(type, access, stages, usage_flags, samples)
    ispow2(samples) || error("The number of samples must be a power of two.")
    new(type, access, stages, usage_flags, samples)
  end
end

combine(x::ImageUsage, y::ImageUsage) = ImageUsage(x.type | y.type, x.access | y.access, x.stages | y.stages, x.usage_flags | y.usage_flags, x.samples | y.samples)

Base.@kwdef struct AttachmentUsage
  type::ShaderResourceType = ShaderResourceType(0)
  access::MemoryAccess = MemoryAccess(0)
  stages::Vk.PipelineStageFlag2 = Vk.PIPELINE_STAGE_2_NONE
  usage_flags::Vk.ImageUsageFlag = Vk.ImageUsageFlag(0)
  aspect::Vk.ImageAspectFlag = Vk.ImageAspectFlag(0) # can be deduced
  samples::Int64 = 1
  clear_value::Optional{NTuple{4,Float32}} = nothing
  resolve_mode::Vk.ResolveModeFlag = Vk.RESOLVE_MODE_AVERAGE_BIT
  function AttachmentUsage(type, access, stages, usage_flags, aspect, samples, clear_value, resolve_mode)
    ispow2(samples) || error("The number of samples must be a power of two.")
    new(type, access, stages, usage_flags, aspect, samples, clear_value, resolve_mode)
  end
end

function Base.merge(x::AttachmentUsage, y::AttachmentUsage)
  x.clear_value === y.clear_value || error("Different clear values across a single pass are not allowed.")
  x.resolve_mode === y.resolve_mode || error("Different multisampling resolve modes across a single pass are not allowed.")
  combine(x, y)
end

Base.merge(x::T, y::T) where {T <: Union{BufferUsage, ImageUsage}} = combine(x, y)

function combine(x::AttachmentUsage, y::AttachmentUsage)
  AttachmentUsage(x.type | y.type, x.access | y.access, x.stages | y.stages, x.usage_flags | y.usage_flags, x.aspect | y.aspect, x.samples | y.samples, x.clear_value, y.resolve_mode)
end

struct ResourceUsage
  id::ResourceID
  usage::Union{BufferUsage,ImageUsage,AttachmentUsage}
end

function Base.merge(x::ResourceUsage, y::ResourceUsage)
  x.id == y.id || error("Resource uses for different resource IDs cannot be merged.")
  ResourceUsage(x.id, merge(x.usage, y.usage))
end

function combine(x::ResourceUsage, y::ResourceUsage)
  x.id == y.id || error("Resource uses for different resource IDs cannot be combined.")
  ResourceUsage(x.id, combine(x.usage, y.usage))
end

const DEFAULT_CLEAR_VALUE = (0.0f0, 0.0f0, 0.0f0, 0.0f0)

function rendering_info(attachment::Attachment, usage::AttachmentUsage)
  clear = !isnothing(usage.clear_value)
  Vk.RenderingAttachmentInfo(
    image_layout(usage.type, usage.access),
    Vk.IMAGE_LAYOUT_UNDEFINED,
    load_op(usage.access, clear),
    store_op(usage.access),
    Vk.ClearValue(Vk.ClearColorValue(something(usage.clear_value, DEFAULT_CLEAR_VALUE)));
    image_view = attachment.view.handle,
  )
end

function rendering_info(attachment::Attachment, usage::AttachmentUsage, resolve_attachment::Attachment, resolve_usage::AttachmentUsage)
  info = rendering_info(attachment, usage)
  setproperties(info, (; resolve_image_layout = image_layout(resolve_usage.type, resolve_usage.access), usage.resolve_mode, resolve_image_view = resolve_attachment.view.handle))
end

"""
Deduce the Vulkan usage, layout and access flags form a resource given its type and access.

The idea is to reconstruct information like `Vk.ACCESS_COLOR_ATTACHMENT_READ_BIT` and `Vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` from a more decoupled description.
"""
function image_layout(type::ShaderResourceType, access::MemoryAccess)
  @match (type, access) begin
    (&SHADER_RESOURCE_TYPE_COLOR_ATTACHMENT, &READ) => Vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    (&SHADER_RESOURCE_TYPE_COLOR_ATTACHMENT, &WRITE) => Vk.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    (&SHADER_RESOURCE_TYPE_COLOR_ATTACHMENT || &SHADER_RESOURCE_TYPE_IMAGE || SHADER_RESOURCE_TYPE_TEXTURE, &(READ | WRITE)) => Vk.IMAGE_LAYOUT_GENERAL
    (&SHADER_RESOURCE_TYPE_DEPTH_ATTACHMENT, &READ) => Vk.IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL
    (&SHADER_RESOURCE_TYPE_DEPTH_ATTACHMENT, &WRITE) => Vk.IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
    (&SHADER_RESOURCE_TYPE_STENCIL_ATTACHMENT, &READ) => Vk.IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL
    (&SHADER_RESOURCE_TYPE_STENCIL_ATTACHMENT, &WRITE) => Vk.IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL
    (&(SHADER_RESOURCE_TYPE_DEPTH_ATTACHMENT | SHADER_RESOURCE_TYPE_STENCIL_ATTACHMENT), &READ) => Vk.IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
    (&(SHADER_RESOURCE_TYPE_DEPTH_ATTACHMENT | SHADER_RESOURCE_TYPE_STENCIL_ATTACHMENT), &WRITE) => Vk.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    (&SHADER_RESOURCE_TYPE_INPUT_ATTACHMENT || &SHADER_RESOURCE_TYPE_TEXTURE || &SHADER_RESOURCE_TYPE_IMAGE, &READ) => Vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    _ => error("Unsupported combination of type $type and access $access")
  end
end

function buffer_usage_flags(type::ShaderResourceType, access::MemoryAccess)
  bits = Vk.BufferUsageFlag(0)

  SHADER_RESOURCE_TYPE_BUFFER | SHADER_RESOURCE_TYPE_STORAGE in type && (bits |= Vk.BUFFER_USAGE_STORAGE_BUFFER_BIT)
  SHADER_RESOURCE_TYPE_PHYSICAL_BUFFER in type && (bits |= Vk.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
  SHADER_RESOURCE_TYPE_VERTEX_BUFFER in type && (bits |= Vk.BUFFER_USAGE_VERTEX_BUFFER_BIT)
  SHADER_RESOURCE_TYPE_INDEX_BUFFER in type && (bits |= Vk.BUFFER_USAGE_INDEX_BUFFER_BIT)

  bits
end

function image_usage_flags(type::ShaderResourceType, access::MemoryAccess)
  bits = Vk.ImageUsageFlag(0)

  SHADER_RESOURCE_TYPE_COLOR_ATTACHMENT in type && (bits |= Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
  (SHADER_RESOURCE_TYPE_DEPTH_ATTACHMENT in type || SHADER_RESOURCE_TYPE_STENCIL_ATTACHMENT in type) && (bits |= Vk.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
  SHADER_RESOURCE_TYPE_INPUT_ATTACHMENT in type && (bits |= Vk.IMAGE_USAGE_INPUT_ATTACHMENT_BIT)
  SHADER_RESOURCE_TYPE_TEXTURE in type && (bits |= Vk.IMAGE_USAGE_SAMPLED_BIT)
  SHADER_RESOURCE_TYPE_IMAGE in type && WRITE in access && (bits |= Vk.IMAGE_USAGE_STORAGE_BIT)

  bits
end

const SHADER_STAGES = |(
  Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT_KHR,
  Vk.PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT_KHR,
  Vk.PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT_KHR,
  Vk.PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT_KHR,
  Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT_KHR,
  Vk.PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR,
)

function access_flags(type::ShaderResourceType, access::MemoryAccess, stages::Vk.PipelineStageFlag2)
  bits = Vk.AccessFlag2(0)
  SHADER_RESOURCE_TYPE_VERTEX_BUFFER in type && (bits |= Vk.ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT)
  SHADER_RESOURCE_TYPE_INDEX_BUFFER in type && (bits |= Vk.ACCESS_2_INDEX_READ_BIT)
  if SHADER_RESOURCE_TYPE_COLOR_ATTACHMENT in type
    READ in access && (bits |= Vk.ACCESS_2_COLOR_ATTACHMENT_READ_BIT)
    WRITE in access && (bits |= Vk.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT)
  end
  if (SHADER_RESOURCE_TYPE_DEPTH_ATTACHMENT in type || SHADER_RESOURCE_TYPE_STENCIL_ATTACHMENT in type)
    #TODO: support mixed access modes (depth write, stencil read and vice-versa)
    READ in access && (bits |= Vk.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT)
    WRITE in access && (bits |= Vk.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
  end
  SHADER_RESOURCE_TYPE_INPUT_ATTACHMENT in type && (bits |= Vk.ACCESS_2_INPUT_ATTACHMENT_READ_BIT)
  if SHADER_RESOURCE_TYPE_BUFFER in type && !iszero(stages & SHADER_STAGES)
    access == READ && (bits |= Vk.ACCESS_2_UNIFORM_READ_BIT)
    WRITE in access && (bits |= Vk.ACCESS_2_SHADER_WRITE_BIT)
  end
  SHADER_RESOURCE_TYPE_TEXTURE in type && READ in access && (bits |= Vk.ACCESS_2_SHADER_READ_BIT)
  SHADER_RESOURCE_TYPE_TEXTURE in type && WRITE in access && (bits |= Vk.ACCESS_2_SHADER_WRITE_BIT)
  bits
end

function aspect_flags(type::ShaderResourceType)
  bits = Vk.ImageAspectFlag(0)
  SHADER_RESOURCE_TYPE_COLOR_ATTACHMENT in type && (bits |= Vk.IMAGE_ASPECT_COLOR_BIT)
  SHADER_RESOURCE_TYPE_DEPTH_ATTACHMENT in type && (bits |= Vk.IMAGE_ASPECT_DEPTH_BIT)
  SHADER_RESOURCE_TYPE_STENCIL_ATTACHMENT in type && (bits |= Vk.IMAGE_ASPECT_STENCIL_BIT)
  bits
end
