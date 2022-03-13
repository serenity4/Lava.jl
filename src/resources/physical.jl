abstract type PhysicalResource end

struct PhysicalBuffer <: PhysicalResource
  uuid::ResourceUUID
  buffer::Vk.Buffer
  memory::Vk.DeviceMemory
  info::LogicalBuffer
end

PhysicalBuffer(buffer::Buffer, uuid = uuid()) = PhysicalBuffer(uuid, handle(buffer), handle(memory(buffer)), LogicalBuffer(buffer))
PhysicalResource(buffer::Buffer) = PhysicalBuffer(buffer)

struct PhysicalImage <: PhysicalResource
  uuid::ResourceUUID
  image::Vk.Image
  memory::Vk.DeviceMemory
  usage::Vk.ImageUsageFlag
  info::LogicalImage
end

PhysicalImage(image::Image, uuid = uuid()) = PhysicalImage(uuid, handle(image), handle(memory(image)), LogicalImage(image))
PhysicalResource(image::Image) = PhysicalBuffer(image)

struct PhysicalAttachment <: PhysicalResource
  uuid::ResourceUUID
  view::Vk.ImageView
  image::Vk.Image
  memory::Vk.DeviceMemory
  aspect::Vk.ImageAspectFlag
  resolve_image_view::Optional{Vk.ImageView}
  resolve_image::Optional{Vk.Image}
  resolve_image_memory::Optional{Vk.DeviceMemory}
  info::LogicalAttachment
end

function PhysicalAttachment(attachment::Attachment, uuid = uuid())
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
    aspect(attachment),
    resolve_image_view,
    resolve_image,
    memory(resolve_image),
    LogicalAttachment(info),
  )
end
PhysicalResource(attachment::Attachment) = PhysicalBuffer(attachment)

format(attachment::PhysicalAttachment) = attachment.info.format
format(::Nothing) = Vk.FORMAT_UNDEFINED

struct PhysicalResources
  buffers::Dictionary{ResourceUUID,PhysicalBuffer}
  images::Dictionary{ResourceUUID,PhysicalImage}
  attachments::Dictionary{ResourceUUID,PhysicalAttachment}
end

PhysicalResources() = PhysicalResources(Dictionary(), Dictionary(), Dictionary())

new!(pres::PhysicalResources, data) = insert!(pres, uuid(), data)

Base.insert!(pres::PhysicalResources, uuid::ResourceUUID, data::Union{Buffer,Image,Attachment}) = insert!(pres, uuid, PhysicalResource(data, uuid))
Base.insert!(pres::PhysicalResources, uuid::ResourceUUID, buffer::PhysicalBuffer) = insert!(pres.buffers, uuid, buffer)
Base.insert!(pres::PhysicalResources, uuid::ResourceUUID, image::PhysicalImage) = insert!(pres.images, uuid, image)
Base.insert!(pres::PhysicalResources, uuid::ResourceUUID, attachment::PhysicalAttachment) = insert!(pres.attachments, uuid, attachment)
