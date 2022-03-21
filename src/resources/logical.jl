abstract type LogicalResource end

struct LogicalBuffer <: LogicalResource
  uuid::ResourceUUID
  size::Int64
end
LogicalBuffer(uuid::ResourceUUID, buffer::Buffer) = LogicalBuffer(uuid, size(buffer))

struct LogicalImage <: LogicalResource
  uuid::ResourceUUID
  format::Vk.Format
  dims::Vector{Int64}
  mip_levels::Int64
  layers::Int64
end
LogicalImage(uuid::ResourceUUID, format::Vk.Format, dims::Tuple, args...) = LogicalImage(uuid, format, collect(dims), args...)
LogicalImage(uuid::ResourceUUID, image::Image) = LogicalImage(uuid, format(image), dims(image), mip_levels(image), layers(image))
LogicalImage(uuid::ResourceUUID, format::Vk.Format, dims; mip_levels = 1, layers = 1) = LogicalImage(uuid, format, dims, mip_levels, layers)

struct LogicalAttachment <: LogicalResource
  uuid::ResourceUUID
  format::Vk.Format
  # If `nothing`, will inherit dimensions from the rendered area.
  dims::Optional{Vector{Int64}}
  mip_range::UnitRange{Int64}
  layer_range::UnitRange{Int64}
  samples::Int64
  resolve_mode::Vk.ResolveModeFlag
end
LogicalAttachment(uuid, format, dims::Tuple, args...) = LogicalAttachment(uuid, format, collect(dims), args...)

function LogicalAttachment(
  uuid::ResourceUUID,
  format,
  dims = nothing;
  mip_range = 0:0,
  layer_range = 1:1,
  samples = 1,
  resolve_mode = Vk.RESOLVE_MODE_AVERAGE_BIT,
)
  ispow2(samples) || error("The number of samples must be a power of two.")
  LogicalAttachment(uuid, format, dims, mip_range, layer_range, samples, resolve_mode)
end
LogicalAttachment(uuid::ResourceUUID, attachment::Attachment) =
  LogicalAttachment(uuid, format(attachment), dims(attachment), mip_range(attachment), layer_range(attachment), samples(attachment), Vk.RESOLVE_MODE_AVERAGE_BIT)

struct LogicalResources
  buffers::Dictionary{ResourceUUID,LogicalBuffer}
  images::Dictionary{ResourceUUID,LogicalImage}
  attachments::Dictionary{ResourceUUID,LogicalAttachment}
end

LogicalResources() = LogicalResources(Dictionary(), Dictionary(), Dictionary())
