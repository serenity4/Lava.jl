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

Base.merge(x::T, y::T) where {T <: Union{BufferUsage, ImageUsage}} = T(x.type | y.type, x.access | y.access, x.stages | y.stages, x.usage | y.usage)

Base.@kwdef struct AttachmentUsage <: ResourceUsage
  type::ResourceType = ResourceType(0)
  access::MemoryAccess = MemoryAccess(0)
  stages::Vk.PipelineStageFlag2 = Vk.PIPELINE_STAGE_2_NONE
  usage::Vk.ImageUsageFlag = Vk.ImageUsageFlag(0)
  aspect::Vk.ImageAspectFlag = Vk.ImageAspectFlag(0) # can be deduced
  samples::Vk.SampleCountFlag = Vk.SampleCountFlag(0)
  resolve_layout::Optional{Vk.ImageLayout} = nothing # can be deduced
  clear_value::Optional{NTuple{4,Float32}}
end

Base.merge(x::AttachmentUsage, y::AttachmentUsage) = AttachmentUsage(x.type | y.type, x.access | y.access, x.stages | y.stages, x.usage | y.usage, x.aspect | y.aspect, x.samples | y.samples, nothing, nothing)

const DEFAULT_CLEAR_VALUE = (0.0f0, 0.0f0, 0.0f0, 0.0f0)

function rendering_info(attachment::PhysicalAttachment, usage::AttachmentUsage)
  clear = !isnothing(usage.clear_value)
  Vk.RenderingAttachmentInfo(
    usage.layout,
    usage.resolve_image_layout,
    load_op(usage.access, clear),
    store_op(usage.access),
    Vk.ClearValue(Vk.ClearColorValue(something(usage.clear_value, DEFAULT_CLEAR_VALUE))),
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

function Base.merge(uses::ResourceUses, other_uses::ResourceUses...)
  ResourceUses(
    reduce(mergewith(merge), getproperty.(other_uses, :buffers); init = uses.buffers),
    reduce(mergewith(merge), getproperty.(other_uses, :images); init = uses.images),
    reduce(mergewith(merge), getproperty.(other_uses, :attachments); init = uses.attachments),
  )
end
