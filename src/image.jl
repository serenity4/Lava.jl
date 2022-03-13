"""
Image with dimension `N` and memory type `M`.
"""
abstract type Image{N,M<:Memory} <: LavaAbstraction end

struct ImageMetaData
  dims::Vector{Int}
  format::Vk.Format
  samples::Vk.SampleCountFlag
  mip_levels::Int
  layers::Int
  usage::Vk.ImageUsageFlag
end

dim(::Type{<:Image{N}}) where {N} = N
dim(im) = dim(typeof(im))

memory_type(::Type{<:Image{N,M}}) where {N,M} = M

Vk.bind_image_memory(image::Image, memory::Memory) = Vk.bind_image_memory(device(image), image, memory, offset(memory))

struct ImageBlock{N,M} <: Image{N,M}
  handle::Vk.Image
  dims::NTuple{N,Int}
  format::Vk.Format
  samples::Vk.SampleCountFlag
  mip_levels::Int
  layers::Int
  usage::Vk.ImageUsageFlag
  queue_family_indices::Vector{Int8}
  sharing_mode::Vk.SharingMode
  is_linear::Bool
  layout::Ref{Vk.ImageLayout}
  memory::Ref{M}
end

dims(image::ImageBlock) = image.dims
format(image::ImageBlock) = image.format
image_layout(image::ImageBlock) = image.layout[]
image(image::ImageBlock) = image
memory(image::ImageBlock) = image.memory[]
isallocated(image::ImageBlock) = isdefined(image.memory, 1)
samples(image::ImageBlock) = image.samples
is_multisampled(x) = samples(x) ≠ Vk.SAMPLE_COUNT_1_BIT
usage(image::ImageBlock) = image.usage

function Vk.Extent3D(image::Image)
  d = dims(image)
  Vk.Extent3D(d..., ntuple(Returns(1), 3 - length(d))...)
end

Vk.Offset3D(image::ImageBlock) = Vk.Offset3D(0, 0, 0)

vk_handle_type(::Type{ImageBlock}) = Vk.Image

function ImageBlock(device, dims, format, usage;
  queue_family_indices = queue_family_indices(device),
  sharing_mode = Vk.SHARING_MODE_EXCLUSIVE,
  memory_type = MemoryBlock,
  is_linear = false,
  preinitialized = false,
  mip_levels = 1,
  array_layers = 1,
  samples = Vk.SAMPLE_COUNT_1_BIT)
  N = length(dims)
  initial_layout = preinitialized ? Vk.IMAGE_LAYOUT_PREINITIALIZED : Vk.IMAGE_LAYOUT_UNDEFINED
  extent_dims = ones(Int, 3)
  extent_dims[1:N] .= dims
  info = Vk.ImageCreateInfo(
    flag(ImageBlock{N,memory_type}),
    format,
    Vk.Extent3D(extent_dims...),
    mip_levels,
    array_layers,
    samples,
    is_linear ? Vk.IMAGE_TILING_LINEAR : Vk.IMAGE_TILING_OPTIMAL,
    usage,
    sharing_mode,
    queue_family_indices,
    initial_layout,
  )
  code = Vk.get_physical_device_image_format_properties(physical_device(device), info.format, info.image_type, info.tiling, info.usage)
  if iserror(code) && unwrap_error(code).code == Vk.ERROR_FORMAT_NOT_SUPPORTED
    error("Format $format not supported for images with tiling $(info.tiling) and usage $(info.usage)")
  end
  handle = unwrap(create(ImageBlock, device, info))
  ImageBlock{N,memory_type}(
    handle,
    dims,
    format,
    samples,
    mip_levels,
    array_layers,
    usage,
    queue_family_indices,
    sharing_mode,
    is_linear,
    Ref(initial_layout),
    Ref{memory_type}(),
  )
end

function Base.similar(image::ImageBlock; memory_domain = nothing, usage = image.usage, is_linear = image.is_linear)
  similar = ImageBlock(
    device(image),
    dims(image),
    format(image),
    usage;
    image.queue_family_indices,
    image.sharing_mode,
    memory_type = memory_type(image),
    is_linear,
    image.mip_levels,
    array_layers = image.layers,
    image.samples,
  )
  if isallocated(image)
    memory_domain = @something(memory_domain, memory(image).domain)
    allocate!(similar, memory_domain)
  end
  similar
end

function bind!(image::ImageBlock, memory::Memory)
  image.memory[] = memory
  unwrap(Vk.bind_image_memory(image, memory))
  image
end

"""
Allocate a `MemoryBlock` and bind it to provided image.
"""
function allocate!(image::ImageBlock, domain::MemoryDomain)
  _device = device(image)
  reqs = Vk.get_image_memory_requirements(_device, image)
  memory = MemoryBlock(_device, reqs.size, reqs.memory_type_bits, domain)
  bind!(image, memory)
end

"""
View of a resource, such as an image or buffer.
"""
abstract type View{O<:LavaAbstraction} <: LavaAbstraction end

struct ImageView{I<:Image} <: View{I}
  handle::Vk.ImageView
  image::I
  format::Vk.Format
  aspect::Vk.ImageAspectFlag
  mip_range::UnitRange
  layer_range::UnitRange
end

image(view::ImageView) = view.image

vk_handle_type(T::Type{<:ImageView}) = Vk.ImageView

image_type(::Type{<:ImageView{I}}) where {I} = I
image_type(view::ImageView) = image_type(typeof(view))

dim(T::Type{<:ImageView}) = dim(image_type(T))
memory_type(T::Type{<:ImageView}) = memory_type(image_type(T))
format(view::ImageView) = view.format

@forward ImageView.image samples, dims, image_layout, usage, Vk.Offset3D, Vk.Extent3D, isallocated, image

function flag(T::Type{<:Image})
  @match dim(T) begin
    1 => Vk.IMAGE_TYPE_1D
    2 => Vk.IMAGE_TYPE_2D
    3 => Vk.IMAGE_TYPE_3D
  end
end

function flag(T::Type{<:ImageView})
  @match dim(T) begin
    1 => Vk.IMAGE_VIEW_TYPE_1D
    2 => Vk.IMAGE_VIEW_TYPE_2D
    3 => Vk.IMAGE_VIEW_TYPE_3D
  end
end

View(image::Image, args...; kwargs...) = ImageView(image, args...; kwargs...)

const DEFAULT_ASPECT = Vk.IMAGE_ASPECT_COLOR_BIT

default_mip_range(image) = 0:(image.mip_levels)
default_layer_range(image) = 1:(image.layers)

function ImageView(
  image::I;
  view_type = flag(ImageView{I}),
  format = format(image),
  component_mapping = Vk.ComponentMapping(
    Vk.COMPONENT_SWIZZLE_IDENTITY,
    Vk.COMPONENT_SWIZZLE_IDENTITY,
    Vk.COMPONENT_SWIZZLE_IDENTITY,
    Vk.COMPONENT_SWIZZLE_IDENTITY,
  ),
  aspect = DEFAULT_ASPECT,
  mip_range = default_mip_range(image),
  layer_range = default_layer_range(image),
) where {I<:Image}

  info = Vk.ImageViewCreateInfo(
    convert(Vk.Image, image),
    view_type,
    format,
    component_mapping,
    subresource_range(aspect, mip_range, layer_range),
  )
  handle = unwrap(create(ImageView, device(image), info))
  ImageView{I}(handle, image, format, aspect, mip_range, layer_range)
end

subresource_range(aspect::Vk.ImageAspectFlag, mip_range::UnitRange, layer_range::UnitRange) =
  Vk.ImageSubresourceRange(aspect, mip_range.start, mip_range.stop - mip_range.start, layer_range.start - 1, 1 + layer_range.stop - layer_range.start)
subresource_range(view::ImageView) = subresource_range(view.aspect, view.mip_range, view.layer_range)
subresource_range(image::Image) = subresource_range(DEFAULT_ASPECT, default_mip_range(image), default_layer_range(image))

subresource_layers(aspect::Vk.ImageAspectFlag = DEFAULT_ASPECT, mip_range::Integer = 0, layer_range::UnitRange = DEFAULT_LAYER_RANGE) =
  Vk.ImageSubresourceLayers(aspect, mip_range, layer_range.start - 1, 1 + layer_range.stop - layer_range.start)
subresource_layers(view::ImageView) = subresource_layers(view.aspect, first(view.mip_range), view.layer_range)
subresource_layers(image::Image) = subresource_layers(DEFAULT_ASPECT, first(default_mip_range(image)), default_layer_range(image))

"""
Opaque image that comes from the Window System Integration (WSI) as returned by `Vk.get_swapchain_images_khr`.
"""
struct ImageWSI <: Image{2,OpaqueMemory}
  handle::Vk.Image
end
