# # Buffers

"""
Create a new buffer and fill it with data.

Note that if the buffer is device-local, a transfer via staging buffer is required.
In this case, a tuple of `(buffer, state)` is returned.
`state` is an [`ExecutionState`](@ref) whose bound resources **must** be
preserved until the transfer has been completed. See the related documentation for more details.
"""
function buffer(device::Device, data = nothing; memory_domain = MEMORY_DOMAIN_DEVICE, usage = Vk.BufferUsageFlag(0), size = nothing, submission = SubmissionInfo(signal_fence = fence(device)), kwargs...)
  isnothing(size) && isnothing(data) && error("At least one of data or size must be provided.")
  isnothing(size) && (size = sizeof(data))

  memory_domain == MEMORY_DOMAIN_DEVICE && !isnothing(data) && (usage |= Vk.BUFFER_USAGE_TRANSFER_DST_BIT)
  buffer = BufferBlock(device, size; usage, kwargs...)
  allocate!(buffer, memory_domain)
  isnothing(data) && return buffer
  state = copyto!(buffer, data; device, submission)
  isnothing(state) && return buffer
  (buffer, state)
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
  Vk.ImageMemoryBarrier2(image_layout(view_or_image), new_layout, 0, 0, handle(image(view_or_image)), subresource_range(view_or_image))
end

function transition_layout(command_buffer::CommandBuffer, view_or_image::Union{<:Image,<:ImageView}, new_layout)
  Vk.cmd_pipeline_barrier_2(command_buffer,
    Vk.DependencyInfo([], [], [transition_layout_info(view_or_image, new_layout)]),
  )
  image(view_or_image).layout[] = new_layout
end

function transition_layout(device::Device, view_or_image::Union{<:Image,<:ImageView}, new_layout)
  command_buffer = request_command_buffer(device, Vk.QUEUE_TRANSFER_BIT)
  transition_layout(command_buffer, view_or_image, new_layout)
  submit(command_buffer, SubmissionInfo(signal_fence = fence(device)))
end

function Base.collect(@nospecialize(T), image::ImageBlock, device::Device)
  if image.is_linear && Vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT in properties(memory(image))
    isbitstype(T) || error("Image type is not an `isbits` type.")
    bytes = collect(memory(image), prod(dims(image)) * sizeof(T), device)
    data = reinterpret(T, bytes)
    reshape(data, dims(image))
  else
    usage = Vk.IMAGE_USAGE_TRANSFER_DST_BIT
    dst = ImageBlock(device, dims(image), format(image), usage; is_linear = true)
    allocate!(dst, MEMORY_DOMAIN_HOST)
    wait(transfer(device, image, dst; submission = SubmissionInfo(signal_fence = fence(device))))
    collect(T, dst, device)
  end
end

function Base.copyto!(image::Image, data::AbstractArray, device::Device; kwargs...)
  b = buffer(device, data; usage = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT, memory_domain = MEMORY_DOMAIN_HOST)
  transfer(device, b, image; kwargs...)
end

function transfer(device::Device, args...; submission = SubmissionInfo(), kwargs...)
  command_buffer = request_command_buffer(device, Vk.QUEUE_TRANSFER_BIT)
  state = transfer(command_buffer, args...; submission, kwargs...)
  something(state, command_buffer)
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

function image(
  device::Device,
  format::Vk.Format,
  data = nothing;
  memory_domain = MEMORY_DOMAIN_DEVICE,
  optimal_tiling = true,
  usage = Vk.IMAGE_USAGE_SAMPLED_BIT,
  dims = nothing,
  samples = 1,
  layout::Optional{Vk.ImageLayout} = nothing,
  submission = SubmissionInfo(signal_fence = fence(device)),
  image_kwargs...,
)
  isnothing(data) && isnothing(dims) && error("Image dimensions must be specified if no data is provided.")
  isnothing(dims) && (dims = size(data))
  !isnothing(data) && (usage |= Vk.IMAGE_USAGE_TRANSFER_DST_BIT)
  # If optimal tiling is enabled, we'll need to transfer the image regardless.
  img = ImageBlock(device, dims, format, usage; is_linear = !optimal_tiling, samples, image_kwargs...)
  allocate!(img, memory_domain)
  state = if !isnothing(data)
    copyto!(img, data, device; submission)
  end
  !isnothing(layout) && wait(transition_layout(device, img, layout))
  isnothing(state) && return img
  (img, state)
end

# # Attachments

function attachment(
  device::Device,
  format::Vk.Format,
  data = nothing;
  usage = Vk.IMAGE_USAGE_SAMPLED_BIT,
  dims = nothing,
  access::MemoryAccess = READ | WRITE,
  samples = 1,
  aspect = Vk.IMAGE_ASPECT_COLOR_BIT,
  kwargs...
)

  if isnothing(data)
    img = image(device, format, data; usage, samples, dims, kwargs...)
    Attachment(View(img; aspect), access)
  else
    img, state = image(device, format, data; usage, samples, dims, kwargs...)
    attachment = Attachment(View(img; aspect), access)
    attachment, state
  end
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
  wait(transfer(device, src, dst; free_src = true, submission = SubmissionInfo(signal_fence = fence(device))))
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
