abstract type ResourceUsage end

struct BufferUsage <: ResourceUsage
  type::ResourceType
  access::MemoryAccess
  stages::Vk.PipelineStageFlag2
  usage::Vk.BufferUsageFlag
end

struct ImageUsage <: ResourceUsage
  type::ResourceType
  access::MemoryAccess
  stages::Vk.PipelineStageFlag2
  usage::Vk.ImageUsageFlag
  layout::Vk.ImageLayout
end

struct AttachmentUsage <: ResourceUsage
  type::ResourceType
  access::MemoryAccess
  stages::Vk.PipelineStageFlag2
  usage::Vk.ImageUsageFlag
  aspect::Vk.ImageAspectFlag
  samples::Vk.SampleCountFlag
  layout::Vk.ImageLayout
  resolve_layout::Optional{Vk.ImageLayout}
  clear_value::Optional{Vk.ClearValue}
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

Base.insert!(uses::ResourceUses, uuid::ResourceUUID, usage::BufferUsage) = insert!(uses.buffers, uuid, usage)
Base.insert!(uses::ResourceUses, uuid::ResourceUUID, image::ImageUsage) = insert!(uses.buffers, uuid, image)
Base.insert!(uses::ResourceUses, uuid::ResourceUUID, attachment::AttachmentUsage) = insert!(uses.buffers, uuid, attachment)
