struct Image <: LavaAbstraction
  handle::Vk.Image
  dims::Vector{Int64}
  flags::Vk.ImageCreateFlag
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
dimensions(image::Image) = image.dims

Vk.bind_image_memory(image::Image, memory::Memory) = Vk.bind_image_memory(device(image), image, memory, memory.offset)

image_layout(image::Image) = image.layout[]
isallocated(image::Image) = isdefined(image.memory, 1)
Base.eltype(image::Image) = format_type(image.format)

is_multisampled(image::Image) = image.samples > 1

mip_range_all(image::Image) = 1:image.mip_levels
layer_range_all(image::Image) = 1:image.layers
nlayers(image::Image) = image.layers

Vk.Extent3D(image::Image) = Vk.Extent3D(image.dims..., ntuple(Returns(1), 3 - length(image.dims))...)
Vk.Offset3D(::Image) = Vk.Offset3D(0, 0, 0)
samples(img::Image) = img.samples

function image_type(ndims)
  @match ndims begin
    1 => Vk.IMAGE_TYPE_1D
    2 => Vk.IMAGE_TYPE_2D
    3 => Vk.IMAGE_TYPE_3D
  end
end

function default_image_flags(array_layers, dims)
  array_layers == 6 && length(dims) == 2 && return Vk.IMAGE_CREATE_CUBE_COMPATIBLE_BIT
  Vk.ImageCreateFlag()
end

function Image(device, dims, format::Union{Vk.Format, DataType}, usage_flags;
  queue_family_indices = queue_family_indices(device),
  sharing_mode = Vk.SHARING_MODE_EXCLUSIVE,
  is_linear = false,
  preinitialized = false,
  mip_levels = 1,
  array_layers = 1,
  samples = 1,
  flags = default_image_flags(array_layers, dims))

  ispow2(samples) || error("The number of samples for must be a power of two.")
  isa(format, DataType) && (format = Vk.Format(format))

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
    initial_layout;
    flags,
  )
  code = Vk.get_physical_device_image_format_properties(physical_device(device), info.format, info.image_type, info.tiling, info.usage; info.flags)
  if iserror(code) && unwrap_error(code).code == Vk.ERROR_FORMAT_NOT_SUPPORTED
    error(styled"""
    Format {yellow:$format} not supported for images with:
      ⚫ {cyan:$(info.flags)}
      ⚫ {cyan:$(info.tiling)}
      ⚫ {cyan:$(info.usage)}
    """)
  end
  handle = unwrap(create(Image, device, info))
  Image(
    handle,
    dims,
    flags,
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

function Base.similar(image::Image; memory_domain = nothing, flags = image.flags, usage_flags = image.usage_flags, is_linear = image.is_linear, samples = image.samples, dims = image.dims, format = image.format, array_layers = image.layers, mip_levels = image.mip_levels)
  similar = Image(
    device(image),
    dims,
    format,
    usage_flags;
    flags,
    image.queue_family_indices,
    image.sharing_mode,
    is_linear,
    mip_levels,
    array_layers,
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
  type::Vk.ImageViewType
  format::Vk.Format
  component_mapping::Vk.ComponentMapping
  aspect::Vk.ImageAspectFlag
  mip_range::UnitRange
  layer_range::UnitRange
end

vk_handle_type(::Type{ImageView}) = Vk.ImageView

@forward_methods ImageView field = :image Vk.Offset3D Vk.Extent3D image_layout samples dimensions nlayers

aspect_flags(view::ImageView) = view.aspect

function image_view_type(dims, layer_range)
  n = length(dims)
  length(layer_range) == 6 && n == 2 && allequal(dims) && return Vk.IMAGE_VIEW_TYPE_CUBE
  length(layer_range) % 6 == 0 && n == 2 && allequal(dims) && return Vk.IMAGE_VIEW_TYPE_CUBE_ARRAY
  @match n begin
    1 => Vk.IMAGE_VIEW_TYPE_1D
    2 => Vk.IMAGE_VIEW_TYPE_2D
    3 => Vk.IMAGE_VIEW_TYPE_3D
  end
end

const COMPONENT_MAPPING_IDENTITY = Vk.ComponentMapping(Vk.COMPONENT_SWIZZLE_IDENTITY, Vk.COMPONENT_SWIZZLE_IDENTITY, Vk.COMPONENT_SWIZZLE_IDENTITY, Vk.COMPONENT_SWIZZLE_IDENTITY)

function ImageView(
  image::Image;
  format = image.format,
  component_mapping = COMPONENT_MAPPING_IDENTITY,
  aspect = aspect_flags(format),
  mip_range = mip_range_all(image),
  layer_range = layer_range_all(image),
  type = image_view_type(dimensions(image), layer_range),
)

  issubset(mip_range, mip_range_all(image)) || error("Mip range $mip_range is not contained within the mip levels defined for the image.")
  issubset(layer_range, layer_range_all(image)) || error("Layer range $layer_range is not contained within the array layers defined for the image.")
  info = Vk.ImageViewCreateInfo(
    image.handle,
    type,
    format,
    component_mapping,
    subresource_range(aspect, mip_range, layer_range),
  )
  handle = unwrap(create(ImageView, device(image), info))
  ImageView(handle, image, type, format, component_mapping, aspect, mip_range, layer_range)
end

subresource_range(aspect::Vk.ImageAspectFlag, mip_range::UnitRange, layer_range::UnitRange) =
  Vk.ImageSubresourceRange(aspect, mip_range.start - 1, 1 + mip_range.stop - mip_range.start, layer_range.start - 1, 1 + layer_range.stop - layer_range.start)
subresource_range(view::ImageView) = subresource_range(aspect_flags(view), view.mip_range, view.layer_range)
subresource_range(image::Image) = subresource_range(aspect_flags(image), mip_range_all(image), layer_range_all(image))

subresource_layers(aspect::Vk.ImageAspectFlag, mip_range::Integer, layer_range::UnitRange) =
  Vk.ImageSubresourceLayers(aspect, mip_range, layer_range.start - 1, 1 + layer_range.stop - layer_range.start)
subresource_layers(view::ImageView; mip_level = first(view.mip_range), layers = view.layer_range) = subresource_layers(aspect_flags(view), mip_level - 1, layers)
subresource_layers(image::Image; mip_level = 1, layers = layer_range_all(image)) = subresource_layers(aspect_flags(image), mip_level - 1, layers)

subresource(aspect::Vk.ImageAspectFlag, mip_level, layer) = Vk.ImageSubresource(aspect, mip_level - 1, layer - 1)
subresource(image::Image) = subresource(aspect_flags(image), first(mip_range_all(image)), first(layer_range_all(image)))
subresource(view::ImageView) = subresource(aspect_flags(image), first(view.mip_range), first(view.layer_range))

function subresource_layout(device, image::Image; aspect = aspect_flags(image), mip_level = first(mip_range_all(image)), layer = first(layer_range_all(image)))
  Vk.get_image_subresource_layout(device, image, subresource(aspect, mip_level, layer))
end
function subresource_layout(device, view::ImageView; aspect = view.aspect, mip_level = view.mip_range[1], layer = view.layer_range[1])
  subresource_layout(device, view.image; aspect, mip_level, layer)
end

aspect_flags(image::Image) = aspect_flags(image.format)
function aspect_flags(format::Vk.Format)
  @match format begin
    &Vk.FORMAT_D16_UNORM || &Vk.FORMAT_D32_SFLOAT || &Vk.FORMAT_X8_D24_UNORM_PACK32 => Vk.IMAGE_ASPECT_DEPTH_BIT
    &Vk.FORMAT_D16_UNORM_S8_UINT || &Vk.FORMAT_D24_UNORM_S8_UINT || &Vk.FORMAT_D32_SFLOAT_S8_UINT => Vk.IMAGE_ASPECT_DEPTH_BIT | Vk.IMAGE_ASPECT_STENCIL_BIT
    &Vk.FORMAT_S8_UINT => Vk.IMAGE_ASPECT_STENCIL_BIT
    _ => Vk.IMAGE_ASPECT_COLOR_BIT
  end
end

function Base.similar(view::ImageView, new_image::Image; type = view.type, format = view.format, component_mapping = view.component_mapping, aspect = view.aspect, mip_range = view.mip_range, layer_range = view.layer_range)
  ImageView(new_image; type, format, component_mapping, aspect, mip_range, layer_range)
end
