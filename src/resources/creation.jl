# Synchronous operations that wait on asynchronous ones.

function transfer(device::Device, args...; kwargs...)
  submission = SubmissionInfo(signal_fence = fence(device))
  command_buffer = request_command_buffer(device, Vk.QUEUE_TRANSFER_BIT)
  wait(transfer(command_buffer, args...; submission, kwargs...))
end

function transition_layout(device::Device, view_or_image::Union{<:Image,<:ImageView}, new_layout)
  command_buffer = request_command_buffer(device, Vk.QUEUE_TRANSFER_BIT)
  transition_layout(command_buffer, view_or_image, new_layout)
  wait(submit(command_buffer, SubmissionInfo(signal_fence = fence(device))))
end

# # Buffers

"""
Create a new buffer and fill it with data.
"""
function buffer(device::Device, data = nothing; memory_domain = MEMORY_DOMAIN_DEVICE, usage = Vk.BufferUsageFlag(0), size = nothing, submission = SubmissionInfo(signal_fence = fence(device)), kwargs...)
  isnothing(size) && isnothing(data) && error("At least one of data or size must be provided.")
  isnothing(size) && (size = sizeof(data))

  memory_domain == MEMORY_DOMAIN_DEVICE && !isnothing(data) && (usage |= Vk.BUFFER_USAGE_TRANSFER_DST_BIT)
  buffer = BufferBlock(device, size; usage, kwargs...)
  allocate!(buffer, memory_domain)
  isnothing(data) && return buffer
  copyto!(buffer, data; device, submission)
end

function Base.copyto!(buffer::BufferBlock, data; device::Optional{Device} = nothing, submission = SubmissionInfo())
  mem = memory(buffer)
  if Vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT in properties(mem)
    device::Device
    tmp = similar(buffer; memory_domain = MEMORY_DOMAIN_HOST, usage = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT)
    copyto!(tmp, data)
    transfer(device, tmp, buffer; submission)
  elseif Vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT in properties(mem)
    copyto!(mem, data)
  else
    error("Buffer not visible neither to device nor to host (memory properties: $(properties(mem))).")
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
  @assert size(src) == size(dst)
  Vk.cmd_copy_buffer(command_buffer, src, dst, [Vk.BufferCopy(offset(src), offset(dst), size(src))])
  push!(command_buffer.to_preserve, dst)
  push!(free_src ? command_buffer.to_free : command_buffer.to_preserve, src)
  isnothing(submission) && return
  Lava.submit(command_buffer, submission)
end

Base.collect(buffer::Buffer) = collect(memory(buffer), size(buffer))
Base.collect(buffer::Buffer, device::Device) = collect(memory(buffer), size(buffer), device)
Base.collect(::Type{T}, buffer::Buffer, device::Optional{Device} = nothing) where {T} = reinterpret(T, collect(buffer, device))

# # Images

function ensure_layout(command_buffer, image_or_view, layout)
  image_layout(image_or_view) == layout && return
  transition_layout(command_buffer, image_or_view, layout)
end

function transfer(
  command_buffer::CommandBuffer,
  src::Union{<:Image,<:ImageView},
  dst::Union{<:Image,<:ImageView};
  submission::Optional{SubmissionInfo} = nothing,
  free_src = false,
)
  @assert dims(src) == dims(dst)
  ensure_layout(command_buffer, src, Vk.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
  ensure_layout(command_buffer, dst, Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)

  if samples(src) ≠ samples(dst)
    # Resolve the source image into a temporary one, and transfer this one to `dst`.
    aux = similar(src; samples = 1)
    ensure_layout(command_buffer, aux, Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    Vk.cmd_resolve_image_2(command_buffer, Vk.ResolveImageInfo2(C_NULL,
      image(src), image_layout(src),
      image(aux), image_layout(aux),
      [Vk.ImageResolve2(subresource_layers(src), Vk.Offset3D(src), subresource_layers(aux), Vk.Offset3D(aux), Vk.Extent3D(src))]
    ))
    return transfer(command_buffer, aux, dst; submission, free_src = true)
  else
    Vk.cmd_copy_image(command_buffer,
      image(src), image_layout(src),
      image(dst), image_layout(dst),
      [Vk.ImageCopy(subresource_layers(src), Vk.Offset3D(src), subresource_layers(dst), Vk.Offset3D(dst), Vk.Extent3D(src))],
    )
  end


  push!(command_buffer.to_preserve, dst)
  push!(free_src ? command_buffer.to_free : command_buffer.to_preserve, src)
  isnothing(submission) && return
  Lava.submit(command_buffer, submission)
end


function transition_layout_info(view_or_image::Union{<:Image,<:ImageView}, new_layout)
  Vk.ImageMemoryBarrier2(image_layout(view_or_image), new_layout, 0, 0, handle(image(view_or_image)), subresource_range(view_or_image);
    src_stage_mask = Vk.PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    dst_stage_mask = Vk.PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    src_access_mask = Vk.ACCESS_2_MEMORY_READ_BIT | Vk.ACCESS_2_MEMORY_WRITE_BIT,
    dst_access_mask = Vk.ACCESS_2_MEMORY_READ_BIT | Vk.ACCESS_2_MEMORY_WRITE_BIT
  )
end

#TODO: This excessively synchronizes, should be also available as part of the render graph.
function transition_layout(command_buffer::CommandBuffer, view_or_image::Union{<:Image,<:ImageView}, new_layout)
  Vk.cmd_pipeline_barrier_2(command_buffer,
    Vk.DependencyInfo([], [], [transition_layout_info(view_or_image, new_layout)]),
  )
  image(view_or_image).layout[] = new_layout
end

function Base.collect(@nospecialize(T), image::ImageBlock, device::Device)
  if image.is_linear && Vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT in properties(memory(image))
    # Get the data from the host-visible memory directly.
    isbitstype(T) || error("Image type is not an `isbits` type.")
    bytes = collect(memory(image), prod(dims(image)) * sizeof(T), device)
    data = reinterpret(T, bytes)
    reshape(data, dims(image))
  else
    # Transfer the data to an image backed by host-visible memory and collect the new image.
    usage = Vk.IMAGE_USAGE_TRANSFER_DST_BIT
    dst = ImageBlock(device, dims(image), format(image), usage; is_linear = true)
    allocate!(dst, MEMORY_DOMAIN_HOST)
    transfer(device, image, dst)
    collect(T, dst, device)
  end
end

function Base.collect(image::Image, device::Device)
  T = format_type(format(image))
  !isnothing(T) || error("The image element type could not be deduced from the image format $(format(image)). Please provide a type as first argument that matches the format of the image.")
  collect(T, image, device)
end

function Base.copyto!(image::Image, data::AbstractArray, device::Device; kwargs...)
  b = buffer(device, data; usage = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT, memory_domain = MEMORY_DOMAIN_HOST)
  transfer(device, b, image; kwargs...)
  image
end

function Base.copyto!(data::AbstractArray, image::Image, device::Device; kwargs...)
  b = buffer(device, data; usage = Vk.BUFFER_USAGE_TRANSFER_DST_BIT, memory_domain = MEMORY_DOMAIN_HOST)
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
  Vk.cmd_copy_buffer_to_image(command_buffer, buffer, image(view_or_image), Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    [
      Vk.BufferImageCopy(
        offset(buffer),
        dims(view_or_image)...,
        subresource_layers(view_or_image),
        Vk.Offset3D(view_or_image),
        Vk.Extent3D(view_or_image),
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
  Vk.cmd_copy_image_to_buffer(command_buffer, image(view_or_image), Vk.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer,
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

function image(
  device::Device,
  data = nothing;
  format = nothing,
  memory_domain = MEMORY_DOMAIN_DEVICE,
  optimal_tiling = true,
  usage = Vk.IMAGE_USAGE_SAMPLED_BIT,
  dims = nothing,
  samples = 1,
  layout::Optional{Vk.ImageLayout} = nothing,
  submission = SubmissionInfo(signal_fence = fence(device)),
  image_kwargs...,
)
  if isnothing(data)
    isnothing(dims) && error("Image dimensions must be specified when no data is provided.")
    isnothing(format) && error("An image format must be specified when no data is provided.")
  else
    isnothing(dims) && (dims = size(data))
    usage |= Vk.IMAGE_USAGE_TRANSFER_DST_BIT
    isnothing(format) && (format = Lava.format(typeof(data)))
    isnothing(format) && error("No format could be determined from the data. Please provide an image format.")
  end
  # If optimal tiling is enabled, we'll need to transfer the image regardless.
  img = ImageBlock(device, dims, format, usage; is_linear = !optimal_tiling, samples, image_kwargs...)
  allocate!(img, memory_domain)
  if !isnothing(data)
    copyto!(img, data, device; submission)
  end
  !isnothing(layout) && transition_layout(device, img, layout)
  img
end

# # Attachments

function attachment(
  device::Device,
  data = nothing;
  access::MemoryAccess = READ | WRITE,
  aspect = Vk.IMAGE_ASPECT_COLOR_BIT,
  image_kwargs...
)

  img = image(device, data; image_kwargs...)
  Attachment(View(img; aspect), access)
end

# # Memory

function Base.collect(memory::MemoryBlock, size::Integer, device::Device)
  Vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT in properties(memory) && return collect(memory, size)
  device::Device
  src = BufferBlock(device, size; usage = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT)
  bind!(src, memory)

  reqs = Vk.get_buffer_memory_requirements(device, src)
  @assert reqs.size ≤ memory.size
  dst = BufferBlock(device, size; usage = Vk.BUFFER_USAGE_TRANSFER_DST_BIT)
  allocate!(dst, MEMORY_DOMAIN_HOST)
  transfer(device, src, dst; free_src = true)
  collect(dst)
end

function Base.collect(memory::MemoryBlock, size::Integer = size(memory))
  map(memory) do ptr
    arrptr = Libc.malloc(size)
    @ccall memmove(arrptr::Ptr{Cvoid}, ptr::Ptr{Cvoid}, size::Csize_t)::Ptr{Cvoid}
    Base.unsafe_wrap(Array, Ptr{UInt8}(arrptr), (size,); own = true)
  end
end

function Base.copyto!(memory::MemoryBlock, data)
  map(memory) do ptr
    ptrcopy!(ptr, data)
  end
  nothing
end
