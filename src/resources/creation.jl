# Synchronous operations that wait on asynchronous ones.

function transfer(device::Device, args...; submission = nothing, kwargs...)
  command_buffer = request_command_buffer(device, Vk.QUEUE_TRANSFER_BIT)
  ret = transfer(command_buffer, args...; submission, kwargs...)
  isnothing(submission) ? ret : wait(ret)
end

function transition_layout(device::Device, view_or_image::Union{<:Image,<:ImageView}, new_layout)
  command_buffer = request_command_buffer(device, Vk.QUEUE_TRANSFER_BIT)
  transition_layout(command_buffer, view_or_image, new_layout)
  wait(submit(command_buffer, SubmissionInfo(signal_fence = fence(device))))
end

# # Buffers

"""
Create and allocate a new buffer and optionally fill it with data.
"""
function Buffer(device::Device; data = nothing, memory_domain::MemoryDomain = MEMORY_DOMAIN_DEVICE, usage_flags = Vk.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, size = nothing, submission = nothing, queue_family_indices = queue_family_indices(device), sharing_mode = Vk.SHARING_MODE_EXCLUSIVE)
  isnothing(size) && isnothing(data) && error("At least one of data or size must be provided.")
  isnothing(size) && (size = sizeof(data))

  memory_domain == MEMORY_DOMAIN_DEVICE && !isnothing(data) && (usage_flags |= Vk.BUFFER_USAGE_TRANSFER_DST_BIT)
  buffer = Buffer(device, size; usage_flags, queue_family_indices, sharing_mode)
  allocate!(buffer, memory_domain)
  isnothing(data) && return buffer
  Vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT in buffer.memory[].property_flags && (submission = @something(submission, SubmissionInfo(signal_fence = fence(device))))
  copyto!(buffer, data; device, submission)
end

function Base.copyto!(buffer::Buffer, data; device::Optional{Device} = nothing, submission = SubmissionInfo())
  mem = buffer.memory[]
  if Vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT in mem.property_flags
    device::Device
    tmp = similar(buffer; memory_domain = MEMORY_DOMAIN_HOST, usage_flags = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT)
    copyto!(tmp, data)
    transfer(device, tmp, buffer; submission)
  elseif Vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT in mem.property_flags
    copyto!(mem, data)
  else
    error("Buffer not visible neither to device nor to host (memory properties: $(mem.property_flags)).")
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
  @assert src.size == dst.size
  Vk.cmd_copy_buffer(command_buffer, src, dst, [Vk.BufferCopy(src.offset, dst.offset, src.size)])
  push!(command_buffer.to_preserve, dst)
  push!(free_src ? command_buffer.to_free : command_buffer.to_preserve, src)
  isnothing(submission) && return
  Lava.submit(command_buffer, submission)
end

Base.collect(buffer::Buffer) = collect(buffer.memory[], buffer.size)
Base.collect(buffer::Buffer, device::Device) = collect(buffer.memory[], buffer.size, device)
Base.collect(::Type{T}, buffer::Buffer) where {T} = reinterpret(T, collect(buffer))
Base.collect(::Type{T}, buffer::Buffer, device::Device) where {T} = reinterpret(T, collect(buffer, device))

# # Images

function ensure_layout(command_buffer, image_or_view, layout)
  image_layout(image_or_view) == layout && return
  transition_layout(command_buffer, image_or_view, layout)
end

get_image_handle(view::ImageView) = view.image.handle
get_image_handle(image::Image) = image.handle
get_image(view::ImageView) = view.image
get_image(image::Image) = image

function transfer(
  command_buffer::CommandBuffer,
  src::Union{<:Image,<:ImageView},
  dst::Union{<:Image,<:ImageView};
  submission::Optional{SubmissionInfo} = nothing,
  free_src = false,
)
  @assert src.dims == dst.dims
  ensure_layout(command_buffer, src, Vk.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
  ensure_layout(command_buffer, dst, Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
  src_image = get_image(src)
  dst_image = get_image(dst)

  if src_image.samples ≠ dst_image.samples
    # Resolve the source image into a temporary one, and transfer this one to `dst`.
    aux = similar(src; samples = 1)
    ensure_layout(command_buffer, aux, Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    Vk.cmd_resolve_image_2(command_buffer, Vk.ResolveImageInfo2(C_NULL,
      src_image, src_image.layout[],
      aux, aux.layout[],
      [Vk.ImageResolve2(subresource_layers(src), Vk.Offset3D(src_image), subresource_layers(aux), Vk.Offset3D(aux), Vk.Extent3D(src))]
    ))
    return transfer(command_buffer, aux, dst; submission, free_src = true)
  else
    Vk.cmd_copy_image(command_buffer,
      src_image, src.layout[],
      dst_image, dst.layout[],
      [Vk.ImageCopy(subresource_layers(src), Vk.Offset3D(src_image), subresource_layers(dst), Vk.Offset3D(dst_image), Vk.Extent3D(src_image))],
    )
  end


  push!(command_buffer.to_preserve, dst)
  push!(free_src ? command_buffer.to_free : command_buffer.to_preserve, src)
  isnothing(submission) && return
  Lava.submit(command_buffer, submission)
end


function transition_layout_info(view_or_image::Union{<:Image,<:ImageView}, new_layout)
  Vk.ImageMemoryBarrier2(image_layout(view_or_image), new_layout, 0, 0, get_image_handle(view_or_image), subresource_range(view_or_image);
    src_stage_mask = Vk.PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    dst_stage_mask = Vk.PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    src_access_mask = Vk.ACCESS_2_MEMORY_READ_BIT | Vk.ACCESS_2_MEMORY_WRITE_BIT,
    dst_access_mask = Vk.ACCESS_2_MEMORY_READ_BIT | Vk.ACCESS_2_MEMORY_WRITE_BIT
  )
end

function transition_layout(command_buffer::CommandBuffer, view_or_image::Union{<:Image,<:ImageView}, new_layout)
  Vk.cmd_pipeline_barrier_2(command_buffer,
    Vk.DependencyInfo([], [], [transition_layout_info(view_or_image, new_layout)]),
  )
  get_image(view_or_image).layout[] = new_layout
end

function Base.collect(@nospecialize(T), image::Image, device::Device)
  @assert image.is_wsi || isallocated(image)
  if image.is_linear && Vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT in image.memory[].property_flags
    # Get the data from the host-visible memory directly.
    isbitstype(T) || error("Image type is not an `isbits` type.")
    bytes = collect(image.memory[], prod(image.dims) * sizeof(T), device)
    data = reinterpret(T, bytes)
    reshape(data, Tuple(image.dims))
  else
    # Transfer the data to an image backed by host-visible memory and collect the new image.
    usage_flags = Vk.IMAGE_USAGE_TRANSFER_DST_BIT
    dst = Image(device, image.dims, image.format, usage_flags; is_linear = true)
    allocate!(dst, MEMORY_DOMAIN_HOST)
    transfer(device, image, dst; submission = sync_submission(device))
    collect(T, dst, device)
  end
end

function Base.collect(image::Image, device::Device)
  T = format_type(image.format)
  !isnothing(T) || error("The image element type could not be deduced from the image format $(image.format). Please provide a type as first argument that matches the format of the image.")
  collect(T, image, device)
end

function Base.copyto!(image::Image, data::AbstractArray, device::Device; kwargs...)
  b = Buffer(device; data, usage_flags = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT, memory_domain = MEMORY_DOMAIN_HOST)
  transfer(device, b, image; kwargs...)
  image
end

function Base.copyto!(data::AbstractArray, image::Image, device::Device; kwargs...)
  b = Buffer(device; data, usage_flags = Vk.BUFFER_USAGE_TRANSFER_DST_BIT, memory_domain = MEMORY_DOMAIN_HOST)
  transfer(device, image, b; kwargs...)
  data
end

function transfer(
  command_buffer::CommandBuffer,
  buffer::Buffer,
  view_or_image::Union{<:Image,<:ImageView};
  submission::Optional{SubmissionInfo} = nothing,
  free_src = false,
)
  transition_layout(command_buffer, view_or_image, Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
  image = get_image(view_or_image)
  Vk.cmd_copy_buffer_to_image(command_buffer, buffer, get_image_handle(view_or_image), Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    [
      Vk.BufferImageCopy(
        buffer.offset,
        image.dims...,
        subresource_layers(view_or_image),
        Vk.Offset3D(image),
        Vk.Extent3D(image),
      ),
    ])
    push!(command_buffer.to_preserve, view_or_image)
    push!(free_src ? command_buffer.to_free : command_buffer.to_preserve, buffer)
    isnothing(submission) && return
    Lava.submit(command_buffer, submission)
end

function transfer(
  command_buffer::CommandBuffer,
  view_or_image::Union{<:Image,<:ImageView},
  buffer::Buffer;
  submission::Optional{SubmissionInfo} = nothing,
  free_src = false,
)
  transition_layout(command_buffer, view_or_image, Vk.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
  Vk.cmd_copy_image_to_buffer(command_buffer, get_image_handle(view_or_image), Vk.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer,
    [
      Vk.BufferImageCopy(
        offset(buffer),
        dims(view_or_image)...,
        subresource_layers(view_or_image),
        Vk.Offset3D(view_or_image),
        Vk.Extent3D(view_or_image),
      ),
    ])
    push!(command_buffer.to_preserve, buffer)
    push!(free_src ? command_buffer.to_free : command_buffer.to_preserve, view_or_image)
    isnothing(submission) && return
    Lava.submit(command_buffer, submission)
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
  layout::Optional{Vk.ImageLayout} = nothing,
  submission = isnothing(data) ? nothing : SubmissionInfo(signal_fence = fence(device)),
)
  if isnothing(data)
    isnothing(dims) && error("Image dimensions must be specified when no data is provided.")
    isnothing(format) && error("An image format must be specified when no data is provided.")
  else
    isnothing(dims) && (dims = collect(size(data)))
    usage_flags |= Vk.IMAGE_USAGE_TRANSFER_DST_BIT
    isnothing(format) && (format = Lava.format(typeof(data)))
    isnothing(format) && error("No format could be determined from the data. Please provide an image format.")
  end
  # If optimal tiling is enabled, we'll need to transfer the image regardless.
  img = Image(device, dims, format, usage_flags; is_linear = !optimal_tiling, samples, queue_family_indices, sharing_mode, mip_levels, array_layers)
  allocate!(img, memory_domain)
  !isnothing(data) && copyto!(img, data, device; submission)
  !isnothing(layout) && transition_layout(device, img, layout)
  img
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
  aspect = Vk.IMAGE_ASPECT_COLOR_BIT,
  mip_range = nothing,
  layer_range = nothing,
  component_mapping = Vk.ComponentMapping(Vk.COMPONENT_SWIZZLE_IDENTITY, Vk.COMPONENT_SWIZZLE_IDENTITY, Vk.COMPONENT_SWIZZLE_IDENTITY, Vk.COMPONENT_SWIZZLE_IDENTITY),
)

  img = Image(device; data, format, memory_domain, optimal_tiling, usage_flags, dims, samples, queue_family_indices, sharing_mode, mip_levels, array_layers, layout)
  mip_range = @something(mip_range, mip_range_all(img))
  layer_range = @something(layer_range, layer_range_all(img))
  Attachment(ImageView(img; aspect, mip_range, layer_range, component_mapping), access)
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
    @ccall memmove(arrptr::Ptr{Cvoid}, ptr::Ptr{Cvoid}, size::Csize_t)::Ptr{Cvoid}
    Base.unsafe_wrap(Array, Ptr{UInt8}(arrptr), (size,); own = true)
  end
end

function Base.copyto!(memory::Memory, data)
  map(memory) do ptr
    ptrcopy!(ptr, data)
  end
  nothing
end
