abstract type LogicalResource end

struct LogicalBuffer <: LogicalResource
  uuid::ResourceUUID
  size::Int
end

LogicalBuffer(buffer::Buffer, uuid = uuid()) = LogicalBuffer(uuid, size(buffer))
LogicalResource(buffer::Buffer, uuid = uuid()) = LogicalBuffer(buffer, uuid)

struct LogicalImage <: LogicalResource
  uuid::ResourceUUID
  format::Vk.Format
  dims::Vector{Int}
end
LogicalImage(image::Image, uuid = uuid()) = LogicalImage(uuid, format(image), dims(image))
LogicalResource(image::Image, uuid = uuid()) = LogicalImage(image, uuid)

struct LogicalAttachment <: LogicalResource
  uuid::ResourceUUID
  # If `nothing`, will inherit dimensions from the rendered area.
  dims::Optional{Vector{Int}}
  format::Vk.Format
  samples::Int
  resolve_mode::Vk.ResolveModeFlag
end

function LogicalAttachment(
  uuid,
  format,
  dims = nothing;
  aspect = Vk.ImageAspectFlag(0),
  samples = 1,
  resolve_mode = Vk.RESOLVE_MODE_AVERAGE_BIT,
)
  ispow2(samples) || error("The number of samples must be a power of two.")
  LogicalAttachment(uuid, format, dims, usage, aspect, size_unit, samples, resolve_mode)
end
LogicalAttachment(attachment::Attachment, uuid = uuid()) =
  LogicalAttachment(uuid, format(attachment), dims(attachment), aspect(attachment), samples(attachment), Vk.RESOLVE_MODE_AVERAGE_BIT)
LogicalResource(attachment::Attachment, uuid = uuid()) = LogicalAttachment(attachment, uuid)

struct LogicalResources
  buffers::Dictionary{ResourceUUID,LogicalBuffer}
  images::Dictionary{ResourceUUID,LogicalImage}
  attachments::Dictionary{ResourceUUID,LogicalAttachment}
end

LogicalResources() = LogicalResources(Dictionary(), Dictionary(), Dictionary())

new!(lres::LogicalResources, data) = insert!(lres, uuid(), data)

Base.insert!(lres::LogicalResources, uuid::ResourceUUID, data::Union{Buffer,Image,Attachment}) = insert!(lres, uuid, LogicalResource(data, uuid))
Base.insert!(lres::LogicalResources, uuid::ResourceUUID, buffer::LogicalBuffer) = insert!(lres.buffers, uuid, buffer)
Base.insert!(lres::LogicalResources, uuid::ResourceUUID, image::LogicalImage) = insert!(lres.images, uuid, image)
Base.insert!(lres::LogicalResources, uuid::ResourceUUID, attachment::LogicalAttachment) = insert!(lres.attachments, uuid, attachment)

new!(lres, args...) = new!(lres, LogicalResource(args...))

buffer_resource(lres, args...) = new!(lres, LogicalBuffer(args...))
image_resource(lres, args...) = new!(lres, LogicalImage(args...))
attachment_resource(lres, args...) = new!(lres, LogicalAttachment(args...))
