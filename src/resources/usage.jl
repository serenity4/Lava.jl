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
  layout::Vk.ImageLayout = Vk.IMAGE_LAYOUT_UNDEFINED
end

Base.@kwdef struct AttachmentUsage <: ResourceUsage
  type::ResourceType = ResourceType(0)
  access::MemoryAccess = MemoryAccess(0)
  stages::Vk.PipelineStageFlag2 = Vk.PIPELINE_STAGE_2_NONE
  usage::Vk.ImageUsageFlag = Vk.ImageUsageFlag(0)
  aspect::Vk.ImageAspectFlag = Vk.ImageAspectFlag(0)
  samples::Vk.SampleCountFlag = Vk.SampleCountFlag(0)
  layout::Vk.ImageLayout = Vk.IMAGE_LAYOUT_UNDEFINED
  resolve_layout::Optional{Vk.ImageLayout} = nothing
  clear_value::Optional{Vk.ClearValue} = nothing
end

const DEFAULT_CLEAR_VALUE = Vk.ClearValue(Vk.ClearColorValue((0.0f0, 0.0f0, 0.0f0, 0.0f0)))

function rendering_info(attachment::PhysicalAttachment, usage::AttachmentUsage)
  clear = !isnothing(usage.clear_value)
  Vk.RenderingAttachmentInfo(
    usage.layout,
    usage.resolve_image_layout,
    load_op(usage.access, clear),
    store_op(usage.access),
    clear ? usage.clear_value : DEFAULT_CLEAR_VALUE;
    attachment.view,
    attachment.resolve_mode,
    resolve_image_view = @something(attachment.resolve_image_view, empty_handle(Vk.ImageView)),
  )
end

struct ResourceUses
  buffers::Dictionary{UUID,BufferUsage}
  images::Dictionary{UUID,ImageUsage}
  attachments::Dictionary{UUID,AttachmentUsage}
end

ResourceUses() = ResourceUses(Dictionary(), Dictionary(), Dictionary())
