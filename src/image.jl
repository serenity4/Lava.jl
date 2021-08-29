"""
Image with dimension `N` and memory type `M`.
"""
abstract type Image{N,M<:Memory} <: LavaAbstraction end

dim(::Type{<:Image{N}}) where {N} = N
dim(im) = dim(typeof(im))

memory_type(::Type{Image{N,M}}) where {N,M} = M

Vk.bind_image_memory(image::Image, memory::Memory) = Vk.bind_image_memory(device(image), image, memory, offset(memory))

struct ImageBlock{N,M} <: Image{N,M}
    handle::Vk.Image
    dims::NTuple{N,Int}
    format::Vk.Format
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
memory(image::ImageBlock) = image.memory[]
isallocated(image::ImageBlock) = isdefined(image.memory, 1)

vk_handle_type(::Type{ImageBlock}) = Vk.Image

function ImageBlock(device, dims, format, usage;
                    queue_family_indices = queue_family_indices(device),
                    sharing_mode = Vk.SHARING_MODE_EXCLUSIVE,
                    memory_type = MemoryBlock,
                    is_linear = false,
                    initial_layout = Vk.IMAGE_LAYOUT_UNDEFINED,
                    mip_levels = 1,
                    array_layers = 1,
                    samples = Vk.SAMPLE_COUNT_1_BIT)
    N = length(dims)
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
    handle = unwrap(create(ImageBlock, device, info))
    ImageBlock{N,memory_type}(handle, dims, format, mip_levels, array_layers, usage, queue_family_indices, sharing_mode, is_linear, Ref(initial_layout), Ref{memory_type}())
end

function bind!(image::ImageBlock, memory::Memory)::Result{ImageBlock,Vk.VulkanError}
    image.memory[] = memory
    @propagate_errors Vk.bind_image_memory(image, memory)
    image
end

"""
Allocate a `MemoryBlock` and bind it to provided image.
"""
function allocate!(image::ImageBlock, domain::MemoryDomain)::Result{ImageBlock,Vk.VulkanError}
    _device = device(image)
    reqs = Vk.get_image_memory_requirements(_device, image)
    @propagate_errors memory = MemoryBlock(_device, reqs.size, reqs.memory_type_bits, domain)
    @propagate_errors bind!(image, memory)
end

"""
View of a resource, such as an image or buffer.
"""
abstract type View{O<:LavaAbstraction} end

struct ImageView{I<:Image} <: View{I}
    handle::Vk.ImageView
    image::I
    format::Vk.Format
    aspect::Vk.ImageAspectFlag
    mip_range::UnitRange
    layer_range::UnitRange
end

image_type(::Type{<:ImageView{I}}) where {I} = I
image_type(view::ImageView) = image_type(typeof(view))

dim(T::Type{<:ImageView}) = dim(image_type(T))

memory_type(T::Type{<:ImageView}) = memory_type(image_type(T))

vk_handle_type(T::Type{<:ImageView}) = Vk.ImageView

format(view::ImageView) = view.format

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

function ImageView(image::I;
              view_type = flag(ImageView{I}),
              format = format(image),
              component_mapping = Vk.ComponentMapping(
                  Vk.COMPONENT_SWIZZLE_IDENTITY,
                  Vk.COMPONENT_SWIZZLE_IDENTITY,
                  Vk.COMPONENT_SWIZZLE_IDENTITY,
                  Vk.COMPONENT_SWIZZLE_IDENTITY
              ),
              aspect = Vk.IMAGE_ASPECT_COLOR_BIT,
              mip_range = 0:image.mip_levels,
              layer_range = 1:image.layers) where {I<:Image}
    info = Vk.ImageViewCreateInfo(
        convert(Vk.Image, image),
        view_type,
        format,
        component_mapping,
        Vk.ImageSubresourceRange(aspect, mip_range.start, mip_range.stop - mip_range.start, layer_range.start - 1, 1 + layer_range.stop - layer_range.start),
    )
    handle = unwrap(create(ImageView, device(image), info))
    ImageView{I}(handle, image, format, aspect, mip_range, layer_range)
end
