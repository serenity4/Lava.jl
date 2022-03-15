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
end

Base.@kwdef struct AttachmentUsage <: ResourceUsage
  type::ResourceType = ResourceType(0)
  access::MemoryAccess = MemoryAccess(0)
  stages::Vk.PipelineStageFlag2 = Vk.PIPELINE_STAGE_2_NONE
  usage::Vk.ImageUsageFlag = Vk.ImageUsageFlag(0)
  aspect::Vk.ImageAspectFlag = Vk.ImageAspectFlag(0) # can be deduced
  samples::Vk.SampleCountFlag = Vk.SampleCountFlag(0)
  resolve_layout::Optional{Vk.ImageLayout} = nothing # can be deduced
end

function rendering_info(attachment::PhysicalAttachment, usage::AttachmentUsage)
  clear = !isnothing(usage.clear_value)
  Vk.RenderingAttachmentInfo(
    usage.layout,
    usage.resolve_image_layout,
    load_op(usage.access, clear),
    store_op(usage.access),
    Vk.ClearValue(Vk.ClearColorValue(something(usage.clear_value, DEFAULT_CLEAR_VALUE)))
    attachment.view,
    attachment.resolve_mode,
    resolve_image_view = @something(attachment.resolve_image_view, empty_handle(Vk.ImageView)),
  )
end

struct ResourceUses
  buffers::Dictionary{ResourceUUID,BufferUsage}
  images::Dictionary{ResourceUUID,ImageUsage}
  attachments::Dictionary{ResourceUUID,AttachmentUsage}
end

ResourceUses() = ResourceUses(Dictionary(), Dictionary(), Dictionary())

const DEFAULT_CLEAR_VALUE = (0.0f0, 0.0f0, 0.0f0, 0.0f0)

# Punctual use
struct ResourceUse
  type::ResourceType
  access::MemoryAccess
  samples::Int
  clear_value::Optional{Tuple{4,Float32}}
end
ResourceUse(type::ResourceType, access::MemoryAccess; samples = 1, clear_value = DEFAULT_CLEAR_VALUE) =
  ResourceUse(type, access, samples, clear_value)
