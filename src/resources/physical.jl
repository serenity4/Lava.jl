abstract type PhysicalResource end

struct PhysicalBuffer <: PhysicalResource
  uuid::ResourceUUID
  buffer::Vk.Buffer
  memory::Vk.DeviceMemory
  usage::Vk.BufferUsageFlag
  offset::Int
  stride::Int
  size::Int
  info::LogicalBuffer
end

PhysicalBuffer(uuid::ResourceUUID, buffer::Buffer) =
  PhysicalBuffer(
    uuid,
    handle(buffer),
    handle(memory(buffer)),
    usage(buffer),
    offset(buffer),
    stride(buffer),
    size(buffer),
    LogicalBuffer(uuid, buffer),
  )

struct PhysicalImage <: PhysicalResource
  uuid::ResourceUUID
  image::Vk.Image
  memory::Vk.DeviceMemory
  usage::Vk.ImageUsageFlag
  layout::Base.RefValue{Vk.ImageLayout}
  info::LogicalImage
end

PhysicalImage(uuid::ResourceUUID, image::Image) = PhysicalImage(uuid, handle(image), handle(memory(image)), usage(image), image.layout, LogicalImage(uuid, image))

struct PhysicalAttachment <: PhysicalResource
  uuid::ResourceUUID
  view::Vk.ImageView
  image::Vk.Image
  memory::Vk.DeviceMemory
  usage::Vk.ImageUsageFlag
  layout::Base.RefValue{Vk.ImageLayout}
  aspect::Vk.ImageAspectFlag
  resolve_image_view::Optional{Vk.ImageView}
  resolve_image::Optional{Vk.Image}
  resolve_image_memory::Optional{Vk.DeviceMemory}
  info::LogicalAttachment
end

function PhysicalAttachment(uuid::ResourceUUID, attachment::Attachment)
  (; image) = attachment.view
  resolve_image = resolve_image_view = nothing
  if is_multisampled(attachment)
    resolve_image = similar(image)
    resolve_image_view = View(resolve_image)
    resolve_image = PhysicalImage(resolve_image)
  end
  PhysicalAttachment(
    uuid,
    attachment.view,
    image,
    memory(image),
    usage(image),
    image.layout,
    aspect(attachment),
    resolve_image_view,
    resolve_image,
    isnothing(resolve_image) ? nothing : memory(resolve_image),
    LogicalAttachment(uuid, attachment),
  )
end

format(attachment::PhysicalAttachment) = attachment.info.format
format(::Nothing) = Vk.FORMAT_UNDEFINED

struct PhysicalResources
  buffers::Dictionary{ResourceUUID,PhysicalBuffer}
  images::Dictionary{ResourceUUID,PhysicalImage}
  attachments::Dictionary{ResourceUUID,PhysicalAttachment}
end

PhysicalResources() = PhysicalResources(Dictionary(), Dictionary(), Dictionary())
