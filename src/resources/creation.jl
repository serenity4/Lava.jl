# Synchronous operations that wait on asynchronous ones.

function transfer(device::Device, args...; submission = nothing, kwargs...)
  command_buffer = request_command_buffer(device, Vk.QUEUE_TRANSFER_BIT)
  ret = transfer(command_buffer, args...; submission, kwargs...)
  isnothing(submission) ? ret : wait(ret)
end

function transition_layout(device::Device, view_or_image::Union{Image,ImageView}, new_layout)
  command_buffer = request_command_buffer(device, Vk.QUEUE_TRANSFER_BIT)
  transition_layout(command_buffer, view_or_image, new_layout)
  wait(submit!(SubmissionInfo(signal_fence = fence(device)), command_buffer))
end

# # Buffers

"""
Create and allocate a new buffer and optionally fill it with data.
"""
function Buffer(device::Device; data = nothing, memory_domain::MemoryDomain = MEMORY_DOMAIN_DEVICE, usage_flags = Vk.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, size = nothing, submission = nothing, queue_family_indices = queue_family_indices(device), sharing_mode = Vk.SHARING_MODE_EXCLUSIVE, layout = NativeLayout())
  isnothing(size) && isnothing(data) && error("At least one of data or size must be provided.")
  isnothing(size) && (size = datasize(layout, data))

  memory_domain == MEMORY_DOMAIN_DEVICE && !isnothing(data) && (usage_flags |= Vk.BUFFER_USAGE_TRANSFER_DST_BIT)
  buffer = Buffer(device, size; usage_flags, queue_family_indices, sharing_mode, layout)
  allocate!(buffer, memory_domain)
  isnothing(data) && return buffer
  Vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT in buffer.memory[].property_flags && (submission = @something(submission, SubmissionInfo(signal_fence = fence(device))))
  copyto!(buffer, data; device, submission)
end

function Base.copyto!(buffer::Buffer, data; device::Optional{Device} = nothing, submission = SubmissionInfo())
  mem = buffer.memory[]
  if !in(Vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT, mem.property_flags)
    device::Device
    tmp = similar(buffer; memory_domain = MEMORY_DOMAIN_HOST, usage_flags = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT, buffer.layout)
    copyto!(tmp, data)
    transfer(device, tmp, buffer; submission)
  else
    copyto!(mem, data, buffer.layout)
  end
  buffer
end

function transfer(
  command_buffer::CommandBuffer,
  src::Buffer,
  dst::Buffer;
  submission::Optional{SubmissionInfo} = nothing,
  free_src = false,
)
  src.size == dst.size || error("Buffer sizes must be equal")
  Vk.cmd_copy_buffer(command_buffer, src, dst, [Vk.BufferCopy(src.offset, dst.offset, src.size)])
  push!(command_buffer.to_preserve, dst)
  push!(free_src ? command_buffer.to_free : command_buffer.to_preserve, src)
  isnothing(submission) && return
  submit!(submission, command_buffer)
end

Base.collect(buffer::Buffer) = collect(buffer.memory[], buffer.size)
Base.collect(buffer::Buffer, device::Device) = collect(buffer.memory[], buffer.size, device)
Base.collect(::Type{T}, buffer::Buffer) where {T} = deserialize(Vector{T}, collect(buffer), buffer.layout)
Base.collect(::Type{T}, buffer::Buffer, device::Device) where {T} = deserialize(Vector{T}, collect(buffer, device), buffer.layout)

# # Images

function ensure_layout(command_buffer, image_or_view, layout)
  image_layout(image_or_view) == layout && return
  transition_layout(command_buffer, image_or_view, layout)
end

get_image_handle(view::ImageView) = view.image.handle
get_image_handle(image::Image) = image.handle
get_image(view::ImageView) = view.image
get_image(image::Image) = image

# From Nicol Bolas: You cannot directly copy depth data into a color image. You can copy the depth data to a buffer via vkCmdCopyImageToBuffer, then copy that data into an image with vkCmdCopyBufferToImage.
# XXX: Implement that logic for such transfers.
function transfer(
  command_buffer::CommandBuffer,
  src::Union{Image,ImageView},
  dst::Union{Image,ImageView};
  submission::Optional{SubmissionInfo} = nothing,
  free_src = false,
)
  dimensions(src) == dimensions(dst) || error("Image dimensions must be the same")
  ensure_layout(command_buffer, src, Vk.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
  ensure_layout(command_buffer, dst, Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
  src_image = get_image(src)
  dst_image = get_image(dst)

  if src_image.samples ≠ dst_image.samples
    # Resolve the source image into a temporary one, and transfer this one to `dst`.
    aux = similar(src; samples = 1)
    ensure_layout(command_buffer, aux, Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    Vk.cmd_resolve_image_2(command_buffer, Vk.ResolveImageInfo2(C_NULL,
      src_image, image_layout(src),
      aux, image_layout(aux),
      [Vk.ImageResolve2(subresource_layers(src), Vk.Offset3D(src_image), subresource_layers(aux), Vk.Offset3D(aux), Vk.Extent3D(src))]
    ))
    return transfer(command_buffer, aux, dst; submission, free_src = true)
  else
    Vk.cmd_copy_image(command_buffer,
      src_image, image_layout(src),
      dst_image, image_layout(dst),
      [Vk.ImageCopy(subresource_layers(src), Vk.Offset3D(src_image), subresource_layers(dst), Vk.Offset3D(dst_image), Vk.Extent3D(src_image))],
    )
  end

  push!(command_buffer.to_preserve, dst)
  push!(free_src ? command_buffer.to_free : command_buffer.to_preserve, src)
  isnothing(submission) && return
  submit!(submission, command_buffer)
end

function transfer(
  command_buffer::CommandBuffer,
  buffer::Buffer,
  view_or_image::Union{Image,ImageView};
  submission::Optional{SubmissionInfo} = nothing,
  free_src = false,
)
  ensure_layout(command_buffer, view_or_image, Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
  regions = buffer_regions_for_transfer(buffer, view_or_image)
  Vk.cmd_copy_buffer_to_image(command_buffer, buffer, get_image_handle(view_or_image), Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, regions)

  push!(command_buffer.to_preserve, view_or_image)
  push!(free_src ? command_buffer.to_free : command_buffer.to_preserve, buffer)
  isnothing(submission) && return
  submit!(submission, command_buffer)
end

function transfer(
  command_buffer::CommandBuffer,
  view_or_image::Union{Image,ImageView},
  buffer::Buffer;
  submission::Optional{SubmissionInfo} = nothing,
  free_src = false,
)
  ensure_layout(command_buffer, view_or_image, Vk.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
  regions = buffer_regions_for_transfer(buffer, view_or_image)
  Vk.cmd_copy_image_to_buffer(command_buffer, get_image_handle(view_or_image), Vk.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, regions)

  push!(command_buffer.to_preserve, buffer)
  push!(free_src ? command_buffer.to_free : command_buffer.to_preserve, view_or_image)
  isnothing(submission) && return
  submit!(submission, command_buffer)
end

buffer_regions_for_transfer(buffer, view_or_image) = nlayers(view_or_image) == 1 ? [whole_buffer_to_whole_image(buffer, view_or_image)] : buffer_regions_to_image_layers(buffer, view_or_image)

function whole_buffer_to_whole_image(buffer::Buffer, view_or_image::Union{Image,ImageView})
  image = get_image(view_or_image)
  Vk.BufferImageCopy(buffer.offset, image.dims..., subresource_layers(view_or_image), Vk.Offset3D(image), Vk.Extent3D(image))
end

function buffer_regions_to_image_layers(buffer::Buffer, view_or_image::Union{Image,ImageView})
  image = get_image(view_or_image)
  n = nlayers(image)
  regions = Vk.BufferImageCopy[]
  buffer_offset = buffer.offset
  image_offset = Vk.Offset3D(image)
  extent = Vk.Extent3D(image)
  width, height = image.dims
  (layer_size, layer_offset) = @match layout = buffer.layout begin
    ::NoPadding || ::NativeLayout => begin
        T = Vk.format_type(image.format)
        ls = datasize(layout, Matrix{T}, (width, height))
        lo = stride(layout, Vector{Matrix{T}}, (width, height))
        (ls, lo)
      end
    _ => error("Buffer layouts other than `NoPadding` or `NativeLayout` are not supported for copying to multiple image layers, got layout of type $(typeof(layout))")
  end
  @assert layer_offset ≥ layer_size "The computed layer offset should be bigger than the computed layer size, otherwise contents will alias each other from one layer to the next"
  for i in 1:n
    buffer_offset + layer_size ≤ buffer.size || error("Buffer overflow detected while copying buffer data to multiple image layers")
    region = Vk.BufferImageCopy(buffer_offset, width, height, subresource_layers(view_or_image; layers = i:i), image_offset, extent)
    push!(regions, region)
    buffer_offset += layer_offset
  end
  regions
end

function transition_layout_info(view_or_image::Union{Image,ImageView}, new_layout)
  image = get_image(view_or_image)

  # Technically, layout transitions may only affect part of an image.
  # But to simplify tracking image layouts, we will transition the whole image.
  # XXX: It may be that for depth/stencil formats, selecting an aspect makes
  # only part of the image to transition; in this case, we'll need to do some aspect/layout tracking.
  subresource = subresource_range(aspect_flags(view_or_image), mip_range_all(image), layer_range_all(image))

  Vk.ImageMemoryBarrier2(image_layout(view_or_image), new_layout, 0, 0, image.handle, subresource;
    src_stage_mask = Vk.PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    dst_stage_mask = Vk.PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    src_access_mask = Vk.ACCESS_2_MEMORY_READ_BIT | Vk.ACCESS_2_MEMORY_WRITE_BIT,
    dst_access_mask = Vk.ACCESS_2_MEMORY_READ_BIT | Vk.ACCESS_2_MEMORY_WRITE_BIT
  )
end

function transition_layout(command_buffer::CommandBuffer, view_or_image::Union{Image,ImageView}, new_layout)
  Vk.cmd_pipeline_barrier_2(command_buffer,
    Vk.DependencyInfo([], [], [transition_layout_info(view_or_image, new_layout)]),
  )
  get_image(view_or_image).layout[] = new_layout
end

function Base.collect(::Type{T}, image::Image, device::Device) where {T}
  isbitstype(T) || error("The image element type is not an `isbits` type")
  image.is_wsi || isallocated(image) || error("The image must be allocated to be collected")
  if image.is_linear && Vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT in image.memory[].property_flags
    # Get the data from the host-visible memory directly.
    # XXX: Query for image subresource layout first instead of assuming a specific memory layout.
    # Can be done via vkGetImageSubresourceLayout2KHR or vkGetImageSubresourceLayout.
    layout = NoPadding()
    bytes = collect(image.memory[], stride(layout, Matrix{T}) * prod(image.dims), device)
    deserialize(Matrix{T}, bytes, layout, image.dims)
  else
    # Transfer the data to an image backed by host-visible memory and collect the new image.
    usage_flags = Vk.IMAGE_USAGE_TRANSFER_DST_BIT
    dst = Image(device, image.dims, image.format, usage_flags; is_linear = true)
    allocate!(dst, MEMORY_DOMAIN_HOST)
    transfer(device, image, dst; submission = sync_submission(device))
    collect(T, dst, device)
  end
end
Base.collect(image::Image, device::Device) = collect(format_type(image.format), image, device)

function Base.collect(::Type{T}, view::ImageView, device::Device) where {T}
  isbitstype(T) || error("The image element type is not an `isbits` type.")
  length(view.mip_range) == 1 || error("Only one mip level may be collected")
  length(view.layer_range) == 1 || error("Only one image layer may be collected")
  (; image) = view
  image.is_wsi || isallocated(image) || error("The image must be allocated to be collected")
  if image.is_linear
    # Transfer the relevant image region into a buffer.
    # XXX: Query for image subresource layout first instead of assuming a specific memory layout.
    # Can be done via vkGetImageSubresourceLayout2KHR or vkGetImageSubresourceLayout.
    layout = NoPadding()
    buffer_size = stride(layout, Matrix{T}) * prod(image.dims)
    buffer = Buffer(device; size = buffer_size, memory_domain = MEMORY_DOMAIN_HOST_CACHED, usage_flags = Vk.BUFFER_USAGE_TRANSFER_DST_BIT)
    transfer(device, view, buffer; submission = sync_submission(device))
    bytes = collect(buffer.memory[], buffer_size, device)
    deserialize(Matrix{T}, bytes, layout, image.dims)
  else
    # Transfer the data to an image backed by host-visible memory and collect the new image.
    usage_flags = image.usage_flags | Vk.IMAGE_USAGE_TRANSFER_DST_BIT | Vk.IMAGE_USAGE_TRANSFER_SRC_BIT
    dst = similar(image; array_layers = 1, is_linear = true, usage_flags, flags = Vk.ImageCreateFlag())
    transfer(device, view, dst; submission = sync_submission(device))
    collect(T, similar(view, dst; layer_range = 1:1), device)
  end
end
Base.collect(view::ImageView, device::Device) = collect(format_type(view.format), view, device)

Base.collect(::Type{T}, attachment::Attachment, device::Device) where {T} = collect(T, attachment.view.image, device)
Base.collect(::Type{T}, resource::Resource, device::Device) where {T} = collect(T, resource.data, device)
Base.collect(attachment::Attachment, device::Device) = collect(attachment.view.image, device)
Base.collect(resource::Resource, device::Device) = collect(resource.data, device)

function Base.copyto!(view_or_image::Union{Image,ImageView}, data::AbstractArray, device::Device; kwargs...)
  b = Buffer(device; data, usage_flags = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT, memory_domain = MEMORY_DOMAIN_HOST)
  transfer(device, b, view_or_image; kwargs...)
  view_or_image
end

function Base.copyto!(data::AbstractArray, image::Image, device::Device; kwargs...)
  b = Buffer(device; data, usage_flags = Vk.BUFFER_USAGE_TRANSFER_DST_BIT, memory_domain = MEMORY_DOMAIN_HOST)
  transfer(device, image, b; kwargs...)
  data
end

function Image(
  device::Device;
  data = nothing,
  format = nothing,
  memory_domain = MEMORY_DOMAIN_DEVICE,
  optimal_tiling = true,
  usage_flags = Vk.IMAGE_USAGE_SAMPLED_BIT,
  dims = nothing,
  samples = 1,
  queue_family_indices = queue_family_indices(device),
  sharing_mode = Vk.SHARING_MODE_EXCLUSIVE,
  mip_levels = 1,
  array_layers = 1,
  flags = nothing,
  layout::Optional{Vk.ImageLayout} = nothing,
  submission = isnothing(data) ? nothing : SubmissionInfo(signal_fence = fence(device)),
)

  check_data(data, array_layers)
  dims, format = infer_dims_and_format(data, dims, format, array_layers)
  flags = @something(flags, default_image_flags(array_layers, dims))
  !isnothing(data) && (usage_flags |= Vk.IMAGE_USAGE_TRANSFER_DST_BIT)
  # If optimal tiling is enabled, we'll need to transfer the image regardless.
  img = Image(device, dims, format, usage_flags; is_linear = !optimal_tiling, samples, queue_family_indices, sharing_mode, mip_levels, array_layers, flags)
  allocate!(img, memory_domain)
  !isnothing(data) && copyto!(img, data, device; submission)
  !isnothing(layout) && transition_layout(device, img, layout)
  img
end

function check_data(data, array_layers)
  isnothing(data) && return true
  array_layers > 1 || return true
  length(data) == array_layers || error("Mismatch detected between number of array layers ($array_layers) and the length of the provided `data` ($(length(data)))")
  Ts = eltype.(data)
  allequal(Ts) || error("All image array layers must have the same element type, got multiple types $(unique(Ts))")
  true
end

function infer_dims_and_format(data, dims, format, array_layers)
  if isnothing(data)
    isnothing(dims) && error("Image dimensions must be specified when no data is provided.")
    isnothing(format) && error("An image format must be specified when no data is provided.")
  else
    layer = array_layers == 1 ? data : data[1]
    isnothing(dims) && (dims = collect(size(layer)))
    isnothing(format) && (format = Vk.Format(eltype(layer)))
    isnothing(format) && error("No format could be determined from the data. Please provide an image format.")
  end
  dims, format
end

# # Attachments

function Attachment(
  device::Device,
  data = nothing;
  format = nothing,
  memory_domain = MEMORY_DOMAIN_DEVICE,
  optimal_tiling = true,
  usage_flags = Vk.IMAGE_USAGE_SAMPLED_BIT,
  dims = nothing,
  samples = 1,
  queue_family_indices = queue_family_indices(device),
  sharing_mode = Vk.SHARING_MODE_EXCLUSIVE,
  mip_levels = 1,
  array_layers = 1,
  layout::Optional{Vk.ImageLayout} = nothing,
  access::MemoryAccess = READ | WRITE,
  aspect = nothing,
  mip_range = nothing,
  layer_range = nothing,
  image_flags = nothing,
  component_mapping = Vk.ComponentMapping(Vk.COMPONENT_SWIZZLE_IDENTITY, Vk.COMPONENT_SWIZZLE_IDENTITY, Vk.COMPONENT_SWIZZLE_IDENTITY, Vk.COMPONENT_SWIZZLE_IDENTITY),
  submission = isnothing(data) ? nothing : SubmissionInfo(signal_fence = fence(device)),
)

  dims, format = infer_dims_and_format(data, dims, format, array_layers)
  !isnothing(data) && (usage_flags |= Vk.IMAGE_USAGE_TRANSFER_DST_BIT)
  img = Image(device; format, memory_domain, optimal_tiling, usage_flags, dims, samples, queue_family_indices, sharing_mode, mip_levels, array_layers, flags = image_flags)
  aspect = img.format == Vk.FORMAT_UNDEFINED ? Vk.IMAGE_ASPECT_COLOR_BIT : aspect_flags(img.format)
  mip_range = @something(mip_range, mip_range_all(img))
  layer_range = @something(layer_range, layer_range_all(img))
  view = ImageView(img; aspect, mip_range, layer_range, component_mapping)
  !isnothing(data) && copyto!(view, data, device; submission)
  !isnothing(layout) && transition_layout(device, view, layout)
  Attachment(view, access)
end

# # Memory

function Base.collect(memory::Memory, size::Integer, device::Device)
  Vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT in memory.property_flags && return collect(memory, size)
  device::Device
  src = Buffer(device, size; usage_flags = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT)
  bind!(src, memory)

  reqs = Vk.get_buffer_memory_requirements(device, src)
  @assert reqs.size ≤ memory.size
  dst = Buffer(device, size; usage_flags = Vk.BUFFER_USAGE_TRANSFER_DST_BIT)
  allocate!(dst, MEMORY_DOMAIN_HOST)
  transfer(device, src, dst; free_src = true, submission = sync_submission(device))
  collect(dst)
end

function Base.collect(memory::Memory, size::Integer = memory.size)
  map(memory) do ptr
    arrptr = Libc.malloc(size)
    @ccall memcpy(arrptr::Ptr{Cvoid}, ptr::Ptr{Cvoid}, size::Csize_t)::Ptr{Cvoid}
    Base.unsafe_wrap(Array, Ptr{UInt8}(arrptr), (size,); own = true)
  end
end

function Base.copyto!(memory::Memory, data::Vector{UInt8})
  map(ptr -> ptrcopy!(ptr, data), memory)
  nothing
end
function Base.copyto!(memory::Memory, data, layout::LayoutStrategy)
  bytes = serialize(data, layout)
  copyto!(memory, bytes)
end
