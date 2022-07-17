abstract type PhysicalResource end

struct PhysicalBuffer <: PhysicalResource
  uuid::ResourceUUID
  buffer::Vk.Buffer
  memory::Vk.DeviceMemory
  usage::Vk.BufferUsageFlag
  offset::Int64
  stride::Int64
  size::Int64
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

@forward PhysicalImage.info (format, dims, mip_levels, layers)

mip_range(image::Union{Image, PhysicalImage}) = 0:(mip_levels(image))
aspect(::Union{Image, PhysicalImage}) = Vk.IMAGE_ASPECT_COLOR_BIT
layer_range(image::Union{Image, PhysicalImage}) = 1:(layers(image))

PhysicalImage(uuid::ResourceUUID, image::Image) = PhysicalImage(uuid, handle(image), handle(memory(image)), usage(image), image.layout, LogicalImage(uuid, image))

function default_view(image::PhysicalImage)
  Vk.ImageView(
    image.image.device,
    image.image,
    image_view_type(length(image.info.dims)),
    format(image),
    Vk.ComponentMapping(
      Vk.COMPONENT_SWIZZLE_IDENTITY,
      Vk.COMPONENT_SWIZZLE_IDENTITY,
      Vk.COMPONENT_SWIZZLE_IDENTITY,
      Vk.COMPONENT_SWIZZLE_IDENTITY,
    ),
    subresource_range(image),
  )
end

struct PhysicalAttachment <: PhysicalResource
  uuid::ResourceUUID
  view::Vk.ImageView
  image::Vk.Image
  memory::Optional{Vk.DeviceMemory}
  usage::Vk.ImageUsageFlag
  layout::Base.RefValue{Vk.ImageLayout}
  aspect::Vk.ImageAspectFlag
  info::LogicalAttachment
end

@forward PhysicalAttachment.info (samples, mip_range, layer_range, format)

dims(attachment::PhysicalAttachment) = Tuple(attachment.info.dims::Vector{Int64})

function PhysicalAttachment(uuid::ResourceUUID, attachment::Attachment)
  (; image) = attachment.view
  PhysicalAttachment(
    uuid,
    attachment.view,
    image,
    memory(image),
    usage(image),
    image.layout,
    aspect(attachment),
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
