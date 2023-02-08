struct Image <: LavaAbstraction
  handle::Vk.Image
  dims::Vector{Int64}
  format::Vk.Format
  samples::Int64
  mip_levels::Int64
  layers::Int64
  usage_flags::Vk.ImageUsageFlag
  queue_family_indices::Vector{Int8}
  sharing_mode::Vk.SharingMode
  is_linear::Bool
  layout::RefValue{Vk.ImageLayout}
  memory::RefValue{Memory}
  # Whether the image comes from the Window System Integration (WSI).
  is_wsi::Bool
end

vk_handle_type(::Type{Image}) = Vk.Image

Base.ndims(image::Image) = length(image.dims)

Vk.bind_image_memory(image::Image, memory::Memory) = Vk.bind_image_memory(device(image), image, memory, memory.offset)

image_layout(image::Image) = image.layout[]
isallocated(image::Image) = isdefined(image.memory, 1)
Base.eltype(image::Image) = format_type(image.format)

is_multisampled(image::Image) = image.samples > 1

mip_range_all(image::Image) = 1:image.mip_levels
layer_range_all(image::Image) = 1:image.layers

Vk.Extent3D(image::Image) = Vk.Extent3D(image.dims..., ntuple(Returns(1), 3 - length(image.dims))...)
Vk.Offset3D(::Image) = Vk.Offset3D(0, 0, 0)

function image_type(ndims)
  @match ndims begin
    1 => Vk.IMAGE_TYPE_1D
    2 => Vk.IMAGE_TYPE_2D
    3 => Vk.IMAGE_TYPE_3D
  end
end

function Image(device, dims, format, usage_flags;
  queue_family_indices = queue_family_indices(device),
  sharing_mode = Vk.SHARING_MODE_EXCLUSIVE,
  is_linear = false,
  preinitialized = false,
  mip_levels = 1,
  array_layers = 1,
  samples = 1)

  n = length(dims)
  initial_layout = preinitialized ? Vk.IMAGE_LAYOUT_PREINITIALIZED : Vk.IMAGE_LAYOUT_UNDEFINED
  extent_dims = ones(Int64, 3)
  extent_dims[1:n] .= dims
  info = Vk.ImageCreateInfo(
    image_type(n),
    format,
    Vk.Extent3D(extent_dims...),
    mip_levels,
    array_layers,
    Vk.SampleCountFlag(samples),
    is_linear ? Vk.IMAGE_TILING_LINEAR : Vk.IMAGE_TILING_OPTIMAL,
    usage_flags,
    sharing_mode,
    queue_family_indices,
    initial_layout,
  )
  code = Vk.get_physical_device_image_format_properties(physical_device(device), info.format, info.image_type, info.tiling, info.usage)
  if iserror(code) && unwrap_error(code).code == Vk.ERROR_FORMAT_NOT_SUPPORTED
    error("Format $format not supported for images with tiling $(info.tiling) and usage bits $(info.usage)")
  end
  handle = unwrap(create(Image, device, info))
  Image(
    handle,
    dims,
    format,
    samples,
    mip_levels,
    array_layers,
    usage_flags,
    queue_family_indices,
    sharing_mode,
    is_linear,
    Ref(initial_layout),
    Ref{Memory}(),
    false,
  )
end

function Base.similar(image::Image; memory_domain = nothing, usage_flags = image.usage_flags, is_linear = image.is_linear, samples = image.samples)
  similar = Image(
    device(image),
    image.dims,
    image.format,
    usage_flags;
    image.queue_family_indices,
    image.sharing_mode,
    is_linear,
    image.mip_levels,
    array_layers = image.layers,
    samples,
  )
  if isallocated(image)
    memory_domain = @something(memory_domain, image.memory[].domain)
    allocate!(similar, memory_domain)
  end
  similar
end

function bind!(image::Image, memory::Memory)
  !isdefined(image.memory, 1) || error("Images can't be bound to memory twice.")
  image.memory[] = memory
  unwrap(Vk.bind_image_memory(image, memory))
  image
end

"""
Allocate memory and bind it to provided image.
"""
function allocate!(image::Image, domain::MemoryDomain)
  !isallocated(image) || error("Can't allocate memory for an image more than once.")
  device = Lava.device(image)
  reqs = Vk.get_image_memory_requirements(device, image)
  memory = Memory(device, reqs.size, reqs.memory_type_bits, domain)
  bind!(image, memory)
end

struct ImageView <: LavaAbstraction
  handle::Vk.ImageView
  image::Image
  format::Vk.Format
  aspect::Vk.ImageAspectFlag
  mip_range::UnitRange
  layer_range::UnitRange
end

vk_handle_type(::Type{ImageView}) = Vk.ImageView

@forward ImageView.image (Vk.Offset3D, Vk.Extent3D, image_layout)

function image_view_type(ndims)
  @match ndims begin
    1 => Vk.IMAGE_VIEW_TYPE_1D
    2 => Vk.IMAGE_VIEW_TYPE_2D
    3 => Vk.IMAGE_VIEW_TYPE_3D
  end
end

function ImageView(
  image::Image;
  view_type = image_view_type(ndims(image)),
  format = image.format,
  component_mapping = Vk.ComponentMapping(
    Vk.COMPONENT_SWIZZLE_IDENTITY,
    Vk.COMPONENT_SWIZZLE_IDENTITY,
    Vk.COMPONENT_SWIZZLE_IDENTITY,
    Vk.COMPONENT_SWIZZLE_IDENTITY,
  ),
  aspect = aspect_flags(format),
  mip_range = mip_range_all(image),
  layer_range = layer_range_all(image),
)

  issubset(mip_range, mip_range_all(image)) || error("Mip range $mip_range is not contained within the mip levels defined for the image.")
  issubset(layer_range, layer_range_all(image)) || error("Layer range $layer_range is not contained within the array layers defined for the image.")
  info = Vk.ImageViewCreateInfo(
    image.handle,
    view_type,
    format,
    component_mapping,
    subresource_range(aspect, mip_range, layer_range),
  )
  handle = unwrap(create(ImageView, device(image), info))
  ImageView(handle, image, format, aspect, mip_range, layer_range)
end

subresource_range(aspect::Vk.ImageAspectFlag, mip_range::UnitRange, layer_range::UnitRange) =
  Vk.ImageSubresourceRange(aspect, mip_range.start - 1, 1 + mip_range.stop - mip_range.start, layer_range.start - 1, 1 + layer_range.stop - layer_range.start)
subresource_range(view::ImageView) = subresource_range(view.aspect, view.mip_range, view.layer_range)
subresource_range(image::Image) = subresource_range(aspect_flags(image.format), mip_range_all(image), layer_range_all(image))

subresource_layers(aspect::Vk.ImageAspectFlag, mip_range::Integer, layer_range::UnitRange) =
  Vk.ImageSubresourceLayers(aspect, mip_range, layer_range.start - 1, 1 + layer_range.stop - layer_range.start)
subresource_layers(view::ImageView) = subresource_layers(view.aspect, first(view.mip_range) - 1, view.layer_range)
subresource_layers(image::Image) = subresource_layers(aspect_flags(image.format), image.mip_levels - 1, layer_range_all(image))
