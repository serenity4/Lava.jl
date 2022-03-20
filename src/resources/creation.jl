# # Buffers

"""
Create a new buffer and fill it with data.

Note that if the buffer is device-local, a transfer via staging buffer is required.
In this case, a tuple of `(buffer, state)` is returned.
`state` is an [`ExecutionState`](@ref) whose bound resources **must** be
preserved until the transfer has been completed. See the related documentation for more details.
"""
function buffer(device::Device, data = nothing; memory_domain = MEMORY_DOMAIN_DEVICE, usage = Vk.BufferUsageFlag(0), size = nothing, kwargs...)
  isnothing(size) && isnothing(data) && error("At least one of data or size must be provided.")
  isnothing(size) && (size = sizeof(data))
  usage |= Vk.BUFFER_USAGE_TRANSFER_DST_BIT
  buffer = BufferBlock(device, size; usage, kwargs...)
  allocate!(buffer, memory_domain)
  isnothing(data) && return buffer
  state = copyto!(buffer, data; device)
  isnothing(state) && return buffer
  (buffer, state)
end

function Base.copyto!(buffer::BufferBlock, data; device::Optional{Device} = nothing)
  mem = memory(buffer)
  if Vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT in properties(mem)
    device::Device
    tmp = similar(buffer; memory_domain = MEMORY_DOMAIN_HOST, usage = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT)
    copyto!(tmp, data)
    transfer(device, tmp, buffer; semaphore = Vk.SemaphoreSubmitInfoKHR(Vk.Semaphore(device), 0, 0), free_src = true, signal_fence = true)
  elseif Vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT in properties(mem)
    map(mem) do ptr
      ptrcopy!(ptr, data)
    end
    nothing
  else
    error("Buffer not visible neither to device nor to host (memory properties: $(properties(mem))).")
  end
end

function transfer(
  device::Device,
  src::Buffer,
  dst::Buffer;
  command_buffer = request_command_buffer(device, Vk.QUEUE_TRANSFER_BIT),
  signal_fence = false,
  semaphore = nothing,
  free_src = false,
)
  @assert size(src) == size(dst)

  Vk.cmd_copy_buffer(command_buffer, src, dst, [Vk.BufferCopy(offset(src), offset(dst), size(src))])

  signal_semaphores = []
  !isnothing(semaphore) && push!(signal_semaphores, semaphore)
  info = Vk.SubmitInfo2KHR([], [Vk.CommandBufferSubmitInfoKHR(command_buffer)], signal_semaphores)
  if free_src
    submit(device, command_buffer.queue_family_index, info; signal_fence, semaphore, free_after_completion = [Ref(src)])
  else
    submit(device, command_buffer.queue_family_index, info; signal_fence, semaphore, release_after_completion = [Ref(src)])
  end
end

Base.collect(buffer::Buffer, device::Optional{Device} = nothing) = collect(memory(buffer), size(buffer), device)
Base.collect(::Type{T}, buffer::Buffer, device::Optional{Device} = nothing) where {T} = reinterpret(T, collect(buffer, device))

# # Images

function transfer(
  device::Device,
  src::Union{<:Image,<:ImageView},
  dst::Union{<:Image,<:ImageView};
  command_buffer = request_command_buffer(device, Vk.QUEUE_TRANSFER_BIT),
  signal_fence = true,
  semaphore = nothing,
  free_src = false,
)

  @assert dims(src) == dims(dst)

  if image_layout(src) ≠ Vk.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
    transition_layout(command_buffer, src, Vk.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
  end
  if image_layout(dst) ≠ Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    transition_layout(command_buffer, dst, Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
  end

  Vk.cmd_copy_image(command_buffer,
    image(src), image_layout(src),
    image(dst), image_layout(dst),
    [Vk.ImageCopy(subresource_layers(src), Vk.Offset3D(src), subresource_layers(dst), Vk.Offset3D(dst), Vk.Extent3D(src))],
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


function transition_layout_info(view_or_image::Union{<:Image,<:ImageView}, new_layout)
  Vk.ImageMemoryBarrier2KHR(image_layout(view_or_image), new_layout, 0, 0, handle(image(view_or_image)), subresource_range(view_or_image))
end

function transition_layout(command_buffer::CommandBuffer, view_or_image::Union{<:Image,<:ImageView}, new_layout)
  Vk.cmd_pipeline_barrier_2_khr(command_buffer,
    Vk.DependencyInfoKHR([], [], [transition_layout_info(view_or_image, new_layout)]),
  )
  image(view_or_image).layout[] = new_layout
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
    wait(transfer(device, image, dst))
    collect(T, dst, device)
  end
end

function transfer(device::Device, data::AbstractArray, image::Image; kwargs...)
  b = buffer(device, data; usage = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT, memory_domain = MEMORY_DOMAIN_HOST)
  transfer(device, b, image; kwargs...)
end

function transfer(
  device::Device,
  buffer::Buffer,
  view_or_image::Union{<:Image,<:ImageView};
  command_buffer = request_command_buffer(device, Vk.QUEUE_TRANSFER_BIT),
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
  info = Vk.SubmitInfo2KHR([], [Vk.CommandBufferSubmitInfoKHR(command_buffer)], [])
  release_after_completion = Ref[Ref(view_or_image)]
  free_after_completion = Ref[]
  push!(free_src ? free_after_completion : release_after_completion, Ref(buffer))
  submit(device, command_buffer.queue_family_index, info; signal_fence = true, free_after_completion, release_after_completion)
end

function image(
  device::Device,
  data = nothing;
  format = Vk.FORMAT_UNDEFINED,
  memory_domain = MEMORY_DOMAIN_DEVICE,
  optimal_tiling = true,
  usage = Vk.IMAGE_USAGE_SAMPLED_BIT,
  dims = nothing,
  samples = Vk.SAMPLE_COUNT_1_BIT,
  kwargs...,
)
  isnothing(data) && isnothing(dims) && error("Image dimensions must be specified if no data is provided.")
  isnothing(dims) && (dims = size(data))
  upload_usage = usage | Vk.IMAGE_USAGE_TRANSFER_DST_BIT
  optimal_tiling && (upload_usage |= Vk.IMAGE_USAGE_TRANSFER_SRC_BIT)
  img = ImageBlock(device, dims, format, isnothing(data) ? usage : upload_usage; is_linear = !optimal_tiling, samples, kwargs...)
  allocate!(img, memory_domain)
  isnothing(data) && return img

  state = transfer(device, data, img; free_src = true)
  !optimal_tiling && isnothing(state) && return img
  !optimal_tiling && return (img, state)
  wait(state)

  dst = similar(img; is_linear = false, usage = usage | Vk.IMAGE_USAGE_TRANSFER_DST_BIT)
  (dst, transfer(device, img, dst; free_src = true))
end

# # Attachments

function attachment(
  device::Device,
  data = nothing;
  format = Vk.FORMAT_UNDEFINED,
  usage = Vk.IMAGE_USAGE_SAMPLED_BIT,
  dims = nothing,
  access::MemoryAccess = READ | WRITE,
  samples = Vk.SAMPLE_COUNT_1_BIT,
  aspect = Vk.IMAGE_ASPECT_COLOR_BIT,
)

  if isnothing(data)
    img = image(device, data; format, usage, samples, dims)
    Attachment(View(img; aspect), access)
  else
    img, state = image(device, data; format, usage, samples, dims)
    attachment = Attachment(View(img; aspect), access)
    attachment, state
  end
end

# # Memory

function Base.collect(memory::MemoryBlock, size::Integer, device::Optional{Device} = nothing)
  Vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT in properties(memory) && return collect(memory, size)
  device::Device
  src = BufferBlock(device, size; usage = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT)
  bind!(src, memory)

  reqs = Vk.get_buffer_memory_requirements(device, src)
  @assert reqs.size ≤ memory.size
  dst = BufferBlock(device, size; usage = Vk.BUFFER_USAGE_TRANSFER_DST_BIT)
  allocate!(dst, MEMORY_DOMAIN_HOST)
  wait(transfer(device, src, dst; signal_fence = true, free_src = true))
  collect(dst)
end

function Base.collect(memory::MemoryBlock, size::Integer = size(memory))
  @assert Vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT in properties(memory)
  map(memory) do mapped
    ptr = Libc.malloc(size)
    @ccall memmove(ptr::Ptr{Cvoid}, mapped::Ptr{Cvoid}, size::Csize_t)::Ptr{Cvoid}
    Base.unsafe_wrap(Array, Ptr{UInt8}(ptr), (size,); own = true)
  end
end
