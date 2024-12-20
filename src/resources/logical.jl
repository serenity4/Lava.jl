abstract type LogicalResource end

struct LogicalBuffer <: LogicalResource
  size::Int64
end

struct LogicalImage <: LogicalResource
  format::Vk.Format
  dims::Vector{Int64}
  layers::Int64
  mip_levels::Int64
  samples::Optional{Int64}
end

aspect_flags(image::LogicalImage) = aspect_flags(image.format)
layer_range(image::LogicalImage) = 1:image.layers
mip_range(image::LogicalImage) = 1:image.mip_levels
Subresource(image::LogicalImage) = Subresource(aspect_flags(image), layer_range(image), mip_range(image))
image_format(image::LogicalImage) = image.format

function LogicalImage(format::Union{Vk.Format, DataType}, dims; layers = 1, mip_levels = 1, samples = nothing)
  isa(format, DataType) && (format = Vk.Format(format))
  LogicalImage(format, dims, layers, mip_levels, samples)
end

struct LogicalImageView <: LogicalResource
  image::LogicalImage
  format::Vk.Format
  subresource::Subresource
end

@forward_methods LogicalImageView field = :image samples dimensions
@forward_methods LogicalImageView field = :subresource aspect_flags layer_range mip_range

image_format(view::LogicalImageView) = view.format

function LogicalImageView(image::LogicalImage; format::Optional{Union{Vk.Format, DataType}} = nothing, aspect = nothing, layer_range = 1:1, mip_range = 1:1)
  format = @match format begin
    ::Nothing => image.format
    ::Vk.Format => format
    ::DataType => Vk.Format(format)
  end
  subresource = Subresource(aspect, layer_range, mip_range)
  LogicalImageView(image, format, subresource)
end 

struct LogicalAttachment <: LogicalResource
  format::Vk.Format
  # If `nothing`, will inherit dimensions from the rendered area.
  dims::Optional{Vector{Int64}}
  subresource::Subresource
  samples::Optional{Int64}
end

@forward_methods LogicalAttachment field = :subresource aspect_flags layer_range mip_range

Subresource(attachment::LogicalAttachment) = attachment.subresource
image_format(attachment::LogicalAttachment) = attachment.format

function LogicalAttachment(
  format::Union{Vk.Format, DataType},
  dims = nothing;
  aspect = nothing,
  layer_range = 1:1,
  mip_range = 1:1,
  samples = nothing,
)
  isa(format, DataType) && (format = Vk.Format(format))
  aspect = @something(aspect, aspect_flags(format))
  subresource = Subresource(aspect, layer_range, mip_range)
  LogicalAttachment(format, dims, subresource, samples)
end


image_dimensions(x::Union{LogicalImage,LogicalAttachment}) = x.dims
attachment_dimensions(x::Union{LogicalImage,LogicalAttachment}) = isnothing(image_dimensions(x)) ? nothing : attachment_dimensions(image_dimensions(x), x.subresource)
dimensions(x::LogicalAttachment) = attachment_dimensions(x)
dimensions(x::LogicalImage) = x.dims
samples(x::Union{LogicalImage,LogicalAttachment}) = something(x.samples, 1)

Vk.set_debug_name(data::Union{LogicalBuffer,LogicalImage,LogicalImageView,LogicalAttachment}, name) = nothing
