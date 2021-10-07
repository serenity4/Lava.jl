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
    ImageBlock{N,memory_type}(handle, dims, format, samples, mip_levels, array_layers, usage, queue_family_indices, sharing_mode, is_linear, Ref(initial_layout), Ref{memory_type}())
end

function Base.similar(image::ImageBlock; memory_domain = nothing, usage = image.usage, is_linear = image.is_linear)
    similar = ImageBlock(device(image), dims(image), format(image), usage; image.queue_family_indices, image.sharing_mode, memory_type = memory_type(image), is_linear, image.mip_levels, array_layers = image.layers, image.samples)
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

default_mip_range(image) = 0:image.mip_levels
default_layer_range(image) = 1:image.layers

function ImageView(
        image::I;
        view_type = flag(ImageView{I}),
        format = format(image),
        component_mapping = Vk.ComponentMapping(
            Vk.COMPONENT_SWIZZLE_IDENTITY,
            Vk.COMPONENT_SWIZZLE_IDENTITY,
            Vk.COMPONENT_SWIZZLE_IDENTITY,
            Vk.COMPONENT_SWIZZLE_IDENTITY
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

subresource_range(aspect::Vk.ImageAspectFlag, mip_range::UnitRange, layer_range::UnitRange) = Vk.ImageSubresourceRange(aspect, mip_range.start, mip_range.stop - mip_range.start, layer_range.start - 1, 1 + layer_range.stop - layer_range.start)
subresource_range(view::ImageView) = subresource_range(view.aspect, view.mip_range, view.layer_range)
subresource_range(image::Image) = subresource_range(DEFAULT_ASPECT, default_mip_range(image), default_layer_range(image))

subresource_layers(aspect::Vk.ImageAspectFlag = DEFAULT_ASPECT, mip_range::Integer = 0, layer_range::UnitRange = DEFAULT_LAYER_RANGE) = Vk.ImageSubresourceLayers(aspect, mip_range, layer_range.start - 1, 1 + layer_range.stop - layer_range.start)
subresource_layers(view::ImageView) = subresource_layers(view.aspect, first(view.mip_range), view.layer_range)
subresource_layers(image::Image) = subresource_layers(DEFAULT_ASPECT, first(default_mip_range(image)), default_layer_range(image))

"""
Opaque image that comes from the Window System Integration (WSI) as returned by `Vk.get_swapchain_images_khr`.
"""
struct ImageWSI <: Image{2,OpaqueMemory}
    handle::Vk.Image
end

function transfer(src::Union{<:Image, <:ImageView}, dst::Union{<:Image, <:ImageView}, device; command_buffer = request_command_buffer(device, Vk.QUEUE_TRANSFER_BIT), signal_fence = true, semaphore = nothing, free_src = false)

    @assert dims(src) == dims(dst)

    if image_layout(src) ≠ Vk.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        transition_layout(command_buffer, device, src, Vk.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
    end
    if image_layout(dst) ≠ Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        transition_layout(command_buffer, device, dst, Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    end

    Vk.cmd_copy_image(command_buffer,
        image(src), image_layout(src),
        image(dst), image_layout(dst),
        [Vk.ImageCopy(subresource_layers(src), Vk.Offset3D(src), subresource_layers(dst), Vk.Offset3D(dst), Vk.Extent3D(src))]
    )
    signal_semaphores = []
    !isnothing(semaphore) && push!(signal_semaphores, semaphore)
    info = Vk.SubmitInfo2KHR([], [Vk.CommandBufferSubmitInfoKHR(command_buffer)], signal_semaphores)
    if free_src
        submit(device, command_buffer.queue_family_index, info; signal_fence, semaphore, free_after_completion = [Ref(src)])
    else
        submit(device, command_buffer.queue_family_index, info; signal_fence, semaphore, release_after_completion = [Ref(src)])
    end
end


function transition_layout_info(view_or_image::Union{<:Image, <:ImageView}, new_layout)
    Vk.ImageMemoryBarrier2KHR(image_layout(view_or_image), new_layout, 0, 0, handle(image(view_or_image)), subresource_range(view_or_image))
end

function transition_layout(command_buffer::CommandBuffer, device, view_or_image::Union{<:Image, <:ImageView}, new_layout)
    Vk.cmd_pipeline_barrier_2_khr(command_buffer,
        Vk.DependencyInfoKHR([], [], [transition_layout_info(view_or_image, new_layout)]),
    )
    image(view_or_image).layout[] = new_layout
end

function Base.collect(@nospecialize(T), image::ImageBlock, device)
    if image.is_linear && Vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT in properties(memory(image))
        isbitstype(T) || error("Image type is not an `isbits` type.")
        bytes = collect(memory(image), prod(dims(image)) * sizeof(T), device)
        data = reinterpret(T, bytes)
        reshape(data, dims(image))
    else
        usage = Vk.IMAGE_USAGE_TRANSFER_DST_BIT
        dst = ImageBlock(device, dims(image), format(image), usage; is_linear = true)
        allocate!(dst, MEMORY_DOMAIN_HOST)
        wait(transfer(image, dst, device))
        collect(T, dst, device)
    end
end

function transfer(data::AbstractArray, image::Image, device; kwargs...)
    b = buffer(device, data; usage = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT)
    transfer(b, image, device; kwargs...)
end

function transfer(buffer::Buffer, view_or_image::Union{<:Image, <:ImageView}, device; command_buffer = request_command_buffer(device, Vk.QUEUE_TRANSFER_BIT), free_src = false)
    transition_layout(command_buffer, device, view_or_image, Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    Vk.cmd_copy_buffer_to_image(command_buffer, buffer, image(view_or_image), Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        [Vk.BufferImageCopy(offset(buffer), dims(view_or_image)..., subresource_layers(view_or_image), Vk.Offset3D(view_or_image), Vk.Extent3D(view_or_image))])
    info = Vk.SubmitInfo2KHR([], [Vk.CommandBufferSubmitInfoKHR(command_buffer)], [])
    release_after_completion = Ref[Ref(view_or_image)]
    free_after_completion = Ref[]
    push!(free_src ? free_after_completion : release_after_completion, Ref(buffer))
    submit(device, command_buffer.queue_family_index, info; signal_fence = true, free_after_completion, release_after_completion)
end

function image(device, data::AbstractArray, format::Vk.Format; memory_domain = MEMORY_DOMAIN_DEVICE, optimal_tiling = true, usage = Vk.IMAGE_USAGE_SAMPLED_BIT, kwargs...)
    upload_usage = usage | Vk.IMAGE_USAGE_TRANSFER_DST_BIT
    optimal_tiling && (upload_usage |= Vk.IMAGE_USAGE_TRANSFER_SRC_BIT)
    img = ImageBlock(device, size(data), format, upload_usage; is_linear = !optimal_tiling)
    allocate!(img, memory_domain)
    wait(transfer(data, img, device; free_src = true))
    !optimal_tiling && return img

    dst = similar(img; is_linear = false, usage = usage | Vk.IMAGE_USAGE_TRANSFER_DST_BIT)
    wait(transfer(img, dst, device; free_src = true))
    dst
end
