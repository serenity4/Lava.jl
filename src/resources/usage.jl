abstract type ResourceUsage end

Base.@kwdef struct BufferUsage <: ResourceUsage
  type::ResourceType = ResourceType(0)
  access::MemoryAccess = MemoryAccess(0)
  stages::Vk.PipelineStageFlag2 = Vk.PIPELINE_STAGE_2_NONE
  usage::Vk.BufferUsageFlag = Vk.BufferUsageFlag(0)
end

Base.@kwdef struct ImageUsage <: ResourceUsage
  type::ResourceType = ResourceType(0)
  access::MemoryAccess = MemoryAccess(0)
  stages::Vk.PipelineStageFlag2 = Vk.PIPELINE_STAGE_2_NONE
  usage::Vk.ImageUsageFlag = Vk.ImageUsageFlag(0)
  samples::Int64 = 1
end

Base.merge(x::T, y::T) where {T <: Union{BufferUsage, ImageUsage}} = T(x.type | y.type, x.access | y.access, x.stages | y.stages, x.usage | y.usage, x.samples | y.samples)

Base.@kwdef struct AttachmentUsage <: ResourceUsage
  type::ResourceType = ResourceType(0)
  access::MemoryAccess = MemoryAccess(0)
  stages::Vk.PipelineStageFlag2 = Vk.PIPELINE_STAGE_2_NONE
  usage::Vk.ImageUsageFlag = Vk.ImageUsageFlag(0)
  aspect::Vk.ImageAspectFlag = Vk.ImageAspectFlag(0) # can be deduced
  samples::Int64 = 1
  resolve_layout::Optional{Vk.ImageLayout} = nothing # can be deduced
  clear_value::Optional{NTuple{4,Float32}}
end

Base.merge(x::AttachmentUsage, y::AttachmentUsage) = AttachmentUsage(x.type | y.type, x.access | y.access, x.stages | y.stages, x.usage | y.usage, x.aspect | y.aspect, x.samples | y.samples, nothing, nothing)

const DEFAULT_CLEAR_VALUE = (0.0f0, 0.0f0, 0.0f0, 0.0f0)

function rendering_info(attachment::PhysicalAttachment, usage::AttachmentUsage)
  clear = !isnothing(usage.clear_value)
  Vk.RenderingAttachmentInfo(
    image_layout(usage),
    something(usage.resolve_layout, Vk.IMAGE_LAYOUT_UNDEFINED),
    load_op(usage.access, clear),
    store_op(usage.access),
    Vk.ClearValue(Vk.ClearColorValue(something(usage.clear_value, DEFAULT_CLEAR_VALUE)));
    image_view = attachment.view,
    attachment.info.resolve_mode,
    resolve_image_view = @something(attachment.resolve_image_view, empty_handle(Vk.ImageView)),
  )
end

struct ResourceUses
  buffers::Dictionary{ResourceUUID,BufferUsage}
  images::Dictionary{ResourceUUID,ImageUsage}
  attachments::Dictionary{ResourceUUID,AttachmentUsage}
end

ResourceUses() = ResourceUses(Dictionary(), Dictionary(), Dictionary())

function Base.merge(uses::ResourceUses, other_uses::ResourceUses...)
  ResourceUses(
    reduce(mergewith(merge), getproperty.(other_uses, :buffers); init = uses.buffers),
    reduce(mergewith(merge), getproperty.(other_uses, :images); init = uses.images),
    reduce(mergewith(merge), getproperty.(other_uses, :attachments); init = uses.attachments),
  )
end

"""
Deduce the Vulkan usage, layout and access flags form a resource given its type, stage and access.

The idea is to reconstruct information like `Vk.ACCESS_COLOR_ATTACHMENT_READ_BIT` and `Vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` from a more decoupled description.
"""
function image_layout(type::ResourceType, access::MemoryAccess, stage::Vk.PipelineStageFlag2)
  @match (type, access) begin
    (&RESOURCE_TYPE_COLOR_ATTACHMENT, &READ) => Vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    (&RESOURCE_TYPE_COLOR_ATTACHMENT, &WRITE) => Vk.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    (&RESOURCE_TYPE_COLOR_ATTACHMENT || &RESOURCE_TYPE_IMAGE || RESOURCE_TYPE_TEXTURE, &(READ | WRITE)) => Vk.IMAGE_LAYOUT_GENERAL
    (&RESOURCE_TYPE_DEPTH_ATTACHMENT, &READ) => Vk.IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL
    (&RESOURCE_TYPE_DEPTH_ATTACHMENT, &WRITE) => Vk.IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
    (&RESOURCE_TYPE_STENCIL_ATTACHMENT, &READ) => Vk.IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL
    (&RESOURCE_TYPE_STENCIL_ATTACHMENT, &WRITE) => Vk.IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL
    (&(RESOURCE_TYPE_DEPTH_ATTACHMENT | RESOURCE_TYPE_STENCIL_ATTACHMENT), &READ) => Vk.IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
    (&(RESOURCE_TYPE_DEPTH_ATTACHMENT | RESOURCE_TYPE_STENCIL_ATTACHMENT), &WRITE) => Vk.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    (&RESOURCE_TYPE_INPUT_ATTACHMENT || &RESOURCE_TYPE_TEXTURE || &RESOURCE_TYPE_IMAGE, &READ) => Vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    _ => error("Unsupported combination of type $type and access $access")
  end
end

image_layout(usage::ResourceUsage) = image_layout(usage.type, usage.access, usage.stages)

function buffer_usage_bits(type::ResourceType, access::MemoryAccess)
  bits = Vk.BufferUsageFlag(0)

  RESOURCE_TYPE_BUFFER | RESOURCE_TYPE_STORAGE in type && (bits |= Vk.BUFFER_USAGE_STORAGE_BUFFER_BIT)
  RESOURCE_TYPE_VERTEX_BUFFER in type && (bits |= Vk.BUFFER_USAGE_VERTEX_BUFFER_BIT)
  RESOURCE_TYPE_INDEX_BUFFER in type && (bits |= Vk.BUFFER_USAGE_INDEX_BUFFER_BIT)
  RESOURCE_TYPE_BUFFER in type && access == READ && (bits |= Vk.BUFFER_USAGE_UNIFORM_BUFFER_BIT)

  bits
end

function image_usage_bits(type::ResourceType, access::MemoryAccess)
  bits = Vk.ImageUsageFlag(0)

  RESOURCE_TYPE_COLOR_ATTACHMENT in type && (bits |= Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
  (RESOURCE_TYPE_DEPTH_ATTACHMENT in type || RESOURCE_TYPE_STENCIL_ATTACHMENT in type) && (bits |= Vk.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
  RESOURCE_TYPE_INPUT_ATTACHMENT in type && (bits |= Vk.IMAGE_USAGE_INPUT_ATTACHMENT_BIT)
  RESOURCE_TYPE_TEXTURE in type && (bits |= Vk.IMAGE_USAGE_SAMPLED_BIT)
  RESOURCE_TYPE_IMAGE in type && WRITE in access && (bits |= Vk.IMAGE_USAGE_STORAGE_BIT)

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

function access_bits(type::ResourceType, access::MemoryAccess, stage::Vk.PipelineStageFlag2)
  bits = Vk.AccessFlag2(0)
  RESOURCE_TYPE_VERTEX_BUFFER in type && (bits |= Vk.ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT)
  RESOURCE_TYPE_INDEX_BUFFER in type && (bits |= Vk.ACCESS_2_INDEX_READ_BIT)
  if RESOURCE_TYPE_COLOR_ATTACHMENT in type
    READ in access && (bits |= Vk.ACCESS_2_COLOR_ATTACHMENT_READ_BIT)
    WRITE in access && (bits |= Vk.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT)
  end
  if (RESOURCE_TYPE_DEPTH_ATTACHMENT in type || RESOURCE_TYPE_STENCIL_ATTACHMENT in type)
    #TODO: support mixed access modes (depth write, stencil read and vice-versa)
    READ in access && (bits |= Vk.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT)
    WRITE in access && (bits |= Vk.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
  end
  RESOURCE_TYPE_INPUT_ATTACHMENT in type && (bits |= Vk.ACCESS_2_INPUT_ATTACHMENT_READ_BIT)
  if RESOURCE_TYPE_BUFFER in type && !iszero(stage & SHADER_STAGES)
    access == READ && (bits |= Vk.ACCESS_2_UNIFORM_READ_BIT)
    WRITE in access && (bits |= Vk.ACCESS_2_SHADER_WRITE_BIT)
  end
  RESOURCE_TYPE_TEXTURE in type && READ in access && (bits |= Vk.ACCESS_2_SHADER_READ_BIT)
  RESOURCE_TYPE_TEXTURE in type && WRITE in access && (bits |= Vk.ACCESS_2_SHADER_WRITE_BIT)
  bits
end

access_bits(usage::ResourceUsage) = access_bits(usage.type, usage.access, usage.stages)

function aspect_bits(type::ResourceType)
  bits = Vk.ImageAspectFlag(0)
  RESOURCE_TYPE_COLOR_ATTACHMENT in type && (bits |= Vk.IMAGE_ASPECT_COLOR_BIT)
  RESOURCE_TYPE_DEPTH_ATTACHMENT in type && (bits |= Vk.IMAGE_ASPECT_DEPTH_BIT)
  RESOURCE_TYPE_STENCIL_ATTACHMENT in type && (bits |= Vk.IMAGE_ASPECT_STENCIL_BIT)
  bits
end
