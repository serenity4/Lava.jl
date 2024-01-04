abstract type LogicalResource end

struct LogicalBuffer <: LogicalResource
  size::Int64
end

struct LogicalImage <: LogicalResource
  format::Vk.Format
  dims::Vector{Int64}
  mip_levels::Int64
  layers::Int64
  samples::Optional{Int64}
end

function LogicalImage(format::Union{Vk.Format, DataType}, dims; mip_levels = 1, array_layers = 1, samples = nothing)
  isa(format, DataType) && (format = Vk.Format(format))
  LogicalImage(format, dims, mip_levels, array_layers, samples)
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

function LogicalAttachment(
  format::Union{Vk.Format, DataType},
  dims = nothing;
  mip_range = 1:1,
  layer_range = 1:1,
  aspect = nothing,
  samples = nothing,
)
  isa(format, DataType) && (format = Vk.Format(format))
  aspect = @something(aspect, aspect_flags(format))
  LogicalAttachment(format, dims, mip_range, layer_range, aspect, samples)
end

aspect_flags(attachment::LogicalAttachment) = attachment.aspect

dimensions(x::Union{LogicalImage,LogicalAttachment}) = x.dims
samples(x::Union{LogicalImage,LogicalAttachment}) = something(x.samples, 1)
