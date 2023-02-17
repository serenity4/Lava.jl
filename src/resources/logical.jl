abstract type LogicalResource end

struct LogicalBuffer <: LogicalResource
  size::Int64
end

logical_buffer(size) = logical_resource(RESOURCE_TYPE_BUFFER, LogicalBuffer(size))

struct LogicalImage <: LogicalResource
  format::Vk.Format
  dims::Vector{Int64}
  mip_levels::Int64
  layers::Int64
  samples::Optional{Int64}
end

function logical_image(format::Union{Vk.Format, DataType}, dims; mip_levels = 1, layers = 1, samples = nothing)
  isa(format, DataType) && (format = Lava.format(format))
  logical_resource(RESOURCE_TYPE_IMAGE, LogicalImage(format, dims, mip_levels, layers, samples))
end

struct LogicalAttachment <: LogicalResource
  format::Vk.Format
  # If `nothing`, will inherit dimensions from the rendered area.
  dims::Optional{Vector{Int64}}
  mip_range::UnitRange{Int64}
  layer_range::UnitRange{Int64}
  aspect::Vk.ImageAspectFlag
  samples::Optional{Int64}
end

function logical_attachment(
  format::Union{Vk.Format, DataType},
  dims = nothing;
  mip_range = 1:1,
  layer_range = 1:1,
  aspect = nothing,
  samples = nothing,
)
  isa(format, DataType) && (format = Lava.format(format))
  aspect = @something(aspect, aspect_flags(format))
  logical_resource(RESOURCE_TYPE_ATTACHMENT, LogicalAttachment(format, dims, mip_range, layer_range, aspect, samples))
end

aspect_flags(attachment::LogicalAttachment) = attachment.aspect

dimensions(x::Union{LogicalImage,LogicalAttachment}) = x.dims
samples(x::Union{LogicalImage,LogicalAttachment}) = something(x.samples, 1)
