abstract type LogicalResource end

struct LogicalBuffer <: LogicalResource
  uuid::ResourceUUID
  size::Int
end
LogicalBuffer(uuid::ResourceUUID, buffer::Buffer) = LogicalBuffer(uuid, size(buffer))

struct LogicalImage <: LogicalResource
  uuid::ResourceUUID
  format::Vk.Format
  dims::Vector{Int}
end
LogicalImage(uuid::ResourceUUID, format::Vk.Format, dims::Tuple) = LogicalImage(uuid, format, collect(dims))
LogicalImage(uuid::ResourceUUID, image::Image) = LogicalImage(uuid, format(image), dims(image))

struct LogicalAttachment <: LogicalResource
  uuid::ResourceUUID
  format::Vk.Format
  # If `nothing`, will inherit dimensions from the rendered area.
  dims::Optional{Vector{Int}}
  samples::Int
  resolve_mode::Vk.ResolveModeFlag
end
LogicalAttachment(uuid, format, dims::Tuple, samples, resolve_mode) = LogicalAttachment(uuid, format, collect(dims), samples, resolve_mode)

function LogicalAttachment(
  uuid::ResourceUUID,
  format,
  dims = nothing;
  samples = 1,
  resolve_mode = Vk.RESOLVE_MODE_AVERAGE_BIT,
)
  ispow2(samples) || error("The number of samples must be a power of two.")
  LogicalAttachment(uuid, format, dims, samples, resolve_mode)
end
LogicalAttachment(uuid::ResourceUUID, attachment::Attachment) =
  LogicalAttachment(uuid, format(attachment), dims(attachment), samples(attachment), Vk.RESOLVE_MODE_AVERAGE_BIT)

struct LogicalResources
  buffers::Dictionary{ResourceUUID,LogicalBuffer}
  images::Dictionary{ResourceUUID,LogicalImage}
  attachments::Dictionary{ResourceUUID,LogicalAttachment}
end

LogicalResources() = LogicalResources(Dictionary(), Dictionary(), Dictionary())
