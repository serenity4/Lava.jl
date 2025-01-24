# Synchronous operations that wait on asynchronous ones.

function transfer(device::Device, args...; submission = nothing, kwargs...)
  command_buffer = request_command_buffer(device, Vk.QUEUE_TRANSFER_BIT)
  ret = transfer(command_buffer, args...; submission, kwargs...)
  isnothing(submission) ? ret : wait(ret)
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
  Vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT in buffer.memory[].property_flags && (submission = @something(submission, SubmissionInfo(signal_fence = get_fence!(device))))
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

function ensure_layout(device::Device, view_or_image::Union{Image, ImageView}, layout::Vk.ImageLayout)
  command_buffer = request_command_buffer(device, Vk.QUEUE_TRANSFER_BIT)
  ensure_layout(command_buffer, view_or_image, layout)
  wait(submit!(SubmissionInfo(signal_fence = get_fence!(device)), command_buffer))
end

function ensure_layout(command_buffer::CommandBuffer, view_or_image::Union{Image, ImageView}, layout::Vk.ImageLayout)
  match_subresource(view_or_image) do matched_layer_range, matched_mip_range, matched_layout
    matched_layout == layout && return
    subresource = Subresource(aspect_flags(view_or_image), matched_layer_range, matched_mip_range)
    transition_layout(command_buffer, view_or_image, subresource, matched_layout, layout)
  end
end

function transition_layout(command_buffer::CommandBuffer, view_or_image::Union{Image,ImageView}, subresource::Subresource, old_layout::Vk.ImageLayout, new_layout::Vk.ImageLayout)
  Vk.cmd_pipeline_barrier_2(command_buffer,
    Vk.DependencyInfo([], [], [Vk.ImageMemoryBarrier2(old_layout, new_layout, 0, 0, get_image(view_or_image).handle, Vk.ImageSubresourceRange(subresource);
      src_stage_mask = Vk.PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      dst_stage_mask = Vk.PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      src_access_mask = Vk.ACCESS_2_MEMORY_READ_BIT | Vk.ACCESS_2_MEMORY_WRITE_BIT,
      dst_access_mask = Vk.ACCESS_2_MEMORY_READ_BIT | Vk.ACCESS_2_MEMORY_WRITE_BIT,
    )]),
  )
  update_layout(view_or_image, subresource, new_layout)
  nothing
end

get_image_handle(view::ImageView) = view.image.handle
get_image_handle(image::Image) = image.handle
get_image(view::ImageView) = view.image
get_image(image::Image) = image

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
    samples(src) > 1 && samples(dst) == 1 || error("If sample counts differ between images, the destination image must have a sample count of one")
    tmp = similar(src; samples = 1)
    resolve_multisampled_image(command_buffer, src, tmp)
    return transfer(command_buffer, tmp, dst; submission, free_src = true)
  else
    if aspect_flags(src) ≠ aspect_flags(dst)
      # Copy to a buffer first, direct image copy is not possible.
      size = sizeof(format_type(image_format(src))) * prod(dimensions(src))
      buffer = Buffer(Vk.device(src), size; usage_flags = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT | Vk.BUFFER_USAGE_TRANSFER_DST_BIT, queue_family_indices = queue_family_indices(command_buffer))
      allocate!(buffer, MEMORY_DOMAIN_DEVICE)
      transfer(command_buffer, src, buffer)
      return transfer(command_buffer, buffer, dst; submission, free_src = true)
    end
    match_subresource(src_image.layout, Subresource(src)) do src_matched_layers, src_matched_mip_levels, src_layout
      match_subresource(dst_image.layout, Subresource(dst)) do dst_matched_layers, dst_matched_mip_levels, dst_layout
        src_subresource = Subresource(aspect_flags(src), src_matched_layers, src_matched_mip_levels)
        dst_subresource = Subresource(aspect_flags(dst), dst_matched_layers, dst_matched_mip_levels)
        Vk.cmd_copy_image(command_buffer,
        src_image, src_layout,
        dst_image, dst_layout,
        [Vk.ImageCopy(src_subresource, Vk.Offset3D(src), dst_subresource, Vk.Offset3D(dst), Vk.Extent3D(src))],
      )
      end
    end
  end

  push!(command_buffer.to_preserve, dst)
  push!(free_src ? command_buffer.to_free : command_buffer.to_preserve, src)
  isnothing(submission) && return
  submit!(submission, command_buffer)
end

function resolve_multisampled_image(command_buffer, image::Union{Image, ImageView}, to::Union{Image, ImageView})
  ensure_layout(command_buffer, image, Vk.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
  ensure_layout(command_buffer, to, Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
  Vk.cmd_resolve_image_2(command_buffer, Vk.ResolveImageInfo2(C_NULL,
    get_image(image), image_layout(image),
    get_image(to), image_layout(to),
    [Vk.ImageResolve2(C_NULL, Subresource(image), Vk.Offset3D(image), Subresource(to), Vk.Offset3D(to), Vk.Extent3D(image))]
  ))
end

function transfer(
  command_buffer::CommandBuffer,
  buffer::Buffer,
  view_or_image::Union{Image,ImageView};
  submission::Optional{SubmissionInfo} = nothing,
  free_src = false,
)
  ensure_layout(command_buffer, view_or_image, Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
  regions = buffer_regions_for_image_transfer(buffer, view_or_image)

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
  if samples(view_or_image) > 1
    tmp = similar(view_or_image; samples = 1)
    resolve_multisampled_image(command_buffer, view_or_image, tmp)
    return transfer(command_buffer, tmp, buffer; submission, free_src = true)
  end

  ensure_layout(command_buffer, view_or_image, Vk.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
  regions = buffer_regions_for_image_transfer(buffer, view_or_image)
  Vk.cmd_copy_image_to_buffer(command_buffer, get_image_handle(view_or_image), Vk.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, regions)

  push!(command_buffer.to_preserve, buffer)
  push!(free_src ? command_buffer.to_free : command_buffer.to_preserve, view_or_image)
  isnothing(submission) && return
  submit!(submission, command_buffer)
end

function image_datasize(buffer::Buffer, (width, height), format, aspect)
  T = specialized_format_type(format, aspect)
  datasize(buffer.layout, Matrix{T}, (width, height))::Int
end

function specialized_format_type(format::Vk.Format, aspect::Vk.ImageAspectFlag)
  @match (format, aspect) begin
    (&Vk.FORMAT_D16_UNORM_S8_UINT, &Vk.IMAGE_ASPECT_DEPTH_BIT) => N0f16
    (&Vk.FORMAT_D24_UNORM_S8_UINT, &Vk.IMAGE_ASPECT_DEPTH_BIT) => NTuple{3, UInt8}
    (&Vk.FORMAT_D32_SFLOAT_S8_UINT, &Vk.IMAGE_ASPECT_DEPTH_BIT) => Float32
    (&Vk.FORMAT_D16_UNORM_S8_UINT || &Vk.FORMAT_D24_UNORM_S8_UINT || &Vk.FORMAT_D32_SFLOAT_S8_UINT, &Vk.IMAGE_ASPECT_STENCIL_BIT) => UInt8
    (_, _) => format_type(format)
  end
end

function buffer_regions_for_image_transfer(buffer::Buffer, view_or_image::Union{Image,ImageView})
  image = get_image(view_or_image)
  regions = Vk.BufferImageCopy[]
  (; offset) = buffer
  image_offset = Vk.Offset3D(view_or_image)
  width, height = image_dimensions(view_or_image)
  aspect = aspect_flags(view_or_image)
  format = image_format(view_or_image)
  for layer in layer_range(view_or_image)
    for mip_level in mip_range(view_or_image)
      nx, ny = mip_dimensions((width, height), mip_level)
      size = image_datasize(buffer, (nx, ny), format, aspect)
      subresource = Subresource(aspect, layer:layer, mip_level:mip_level)
      offset + size ≤ buffer.size || error("Buffer overflow detected while copying buffer data to image (offset $offset + mip size $size exceeds buffer size $(buffer.size))")
      region = Vk.BufferImageCopy(offset, nx, ny, subresource, image_offset, extent3d((nx, ny)))
      push!(regions, region)
      offset += size
    end
  end
  regions
end

function Base.collect(::Type{T}, image::Image, device::Device; mip_level = 1, layer = 1) where {T}
  isbitstype(T) || error("The image element type is not an `isbits` type")
  image.is_wsi || isallocated(image) || error("The image must be allocated to be collected")
  if image.is_linear && Vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT in image.memory[].property_flags
    # Get the data from the host-visible memory directly.
    # Since we don't rely on driver transfers to rearrange the data nicely, we have to query and follow the driver-dependent layout.
    (; offset, size, row_pitch) = subresource_layout(device, image, Subresource(aspect_flags(image), layer:layer, mip_level:mip_level))
    memory = @view(image.memory[][offset:(size + offset)])
    bytes = collect(memory, size, device)
    layout = NoPadding()
    column_size = stride(layout, Matrix{T}) * image.dims[1]
    column_padding = Int64(row_pitch) - column_size
    @assert column_padding ≥ 0
    deserialize(Matrix{T}, bytes, layout, image.dims, column_padding)
  else
    # Transfer the image into a buffer, then collect it.
    layout = NoPadding()
    s = stride(layout, Matrix{T})
    width, height = dimensions(image)
    usage_flags = Vk.IMAGE_USAGE_TRANSFER_DST_BIT
    mip_range = @__MODULE__().mip_range(image)
    layer_range = @__MODULE__().layer_range(image)
    size = length(layer_range) * s * sum(i -> prod(mip_dimensions((width, height), i)), mip_range)
    buffer = Buffer(device; size, memory_domain = MEMORY_DOMAIN_HOST_CACHED, usage_flags = Vk.BUFFER_USAGE_TRANSFER_DST_BIT)
    transfer(device, image, buffer; submission = sync_submission(device))
    bytes = collect(buffer, device)
    length(mip_range) == 1 && return deserialize(Matrix{T}, bytes, layout, mip_dimensions((width, height), mip_range[1]))
    mipmaps = Matrix{T}[]
    offset = 0
    for i in mip_range
      nx, ny = mip_dimensions((width, height), i)
      size = image_datasize(buffer, (nx, ny), view.format, aspect)
      region = @view bytes[(offset + 1):(offset + size)]
      push!(mipmaps, deserialize(Matrix{T}, region, layout, (nx, ny)))
      offset += size
    end
    mipmaps
  end
end
Base.collect(image::Image, device::Device; mip_level = 1, layer = 1) = collect(format_type(image.format), image, device; mip_level, layer)

function Base.collect(::Type{T}, view::ImageView, device::Device) where {T}
  (; layer_range, mip_range, aspect) = view.subresource
  length(layer_range) == 1 || error("Only one image layer may be collected")
  (; image) = view
  isbitstype(T) || error("The image element type is not an `isbits` type.")
  image.is_wsi || isallocated(image) || error("The image must be allocated to be collected")
  # Transfer the relevant image region into a buffer.
  layout = NoPadding()
  s = stride(layout, Matrix{T})
  width, height = dimensions(image)
  size = length(layer_range) * s * sum(i -> prod(mip_dimensions((width, height), i)), mip_range)
  buffer = Buffer(device; size, memory_domain = MEMORY_DOMAIN_HOST_CACHED, usage_flags = Vk.BUFFER_USAGE_TRANSFER_DST_BIT)
  transfer(device, view, buffer; submission = sync_submission(device))
  bytes = collect(buffer, device)
  length(mip_range) == 1 && return deserialize(Matrix{T}, bytes, layout, mip_dimensions((width, height), mip_range[1]))
  mipmaps = Matrix{T}[]
  offset = 0
  for i in mip_range
    nx, ny = mip_dimensions((width, height), i)
    size = image_datasize(buffer, (nx, ny), view.format, aspect)
    region = @view bytes[(offset + 1):(offset + size)]
    push!(mipmaps, deserialize(Matrix{T}, region, layout, (nx, ny)))
    offset += size
  end
  mipmaps
end
Base.collect(view::ImageView, device::Device) = collect(format_type(view), view, device)

Base.collect(::Type{T}, attachment::Attachment, device::Device) where {T} = collect(T, attachment.view.image, device)
Base.collect(::Type{T}, resource::Resource, device::Device) where {T} = collect(T, resource.data, device)
Base.collect(attachment::Attachment, device::Device) = collect(attachment.view, device)
Base.collect(resource::Resource, device::Device; kwargs...) = collect(resource.data, device; kwargs...)

function Base.copyto!(view::ImageView, data::AbstractArray, device::Device; submission = SubmissionInfo(signal_fence = get_fence!(device)), kwargs...)
  buffer = Buffer(device; data, usage_flags = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT, memory_domain = MEMORY_DOMAIN_HOST)
  transfer(device, buffer, view; submission, kwargs...)
  view
end

function Base.copyto!(image::Image, data::AbstractArray, device::Device; mip_range = nothing, layer_range = nothing, mip_level = nothing, layer = nothing, submission = SubmissionInfo(signal_fence = get_fence!(device)), kwargs...)
  if !isnothing(mip_level)
    isnothing(mip_range) || throw(ArgumentError("The mip range and mip level must not be provided at the same time"))
    mip_range = mip_level:mip_level
  end
  if !isnothing(layer)
    isnothing(layer_range) || throw(ArgumentError("The layer range and layer must not be provided at the same time"))
    layer_range = layer:layer
  end
  mip_range = something(mip_range, Lava.mip_range(image))
  layer_range = something(layer_range, Lava.layer_range(image))
  view = ImageView(image; mip_range, layer_range)
  copyto!(view, data, device; submission, kwargs...)
  image
end

function Base.copyto!(data::AbstractArray, image::Image, device::Device; kwargs...)
  buffer = Buffer(device; data, usage_flags = Vk.BUFFER_USAGE_TRANSFER_DST_BIT, memory_domain = MEMORY_DOMAIN_HOST)
  transfer(device, image, buffer; kwargs...)
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
  layers = 1,
  mip_levels = 1,
  flags = nothing,
  layout::Optional{Vk.ImageLayout} = nothing,
  submission = isnothing(data) ? nothing : SubmissionInfo(signal_fence = get_fence!(device)),
)

  check_data(data, layers)
  dims, format = infer_dims_and_format(data, dims, format, layers)
  flags = @something(flags, default_image_flags(layers, dims))
  !isnothing(data) && (usage_flags |= Vk.IMAGE_USAGE_TRANSFER_DST_BIT)
  # If optimal tiling is enabled, we'll need to transfer the image regardless.
  img = Image(device, dims, format, usage_flags; is_linear = !optimal_tiling, samples, queue_family_indices, sharing_mode, layers, mip_levels, flags)
  allocate!(img, memory_domain)
  !isnothing(data) && copyto!(img, data, device; submission)
  !isnothing(layout) && ensure_layout(device, img, layout)
  img
end

function check_data(data, layers)
  isnothing(data) && return true
  layers > 1 || return true
  length(data) == layers || error("Mismatch detected between number of array layers ($layers) and the length of the provided `data` ($(length(data)))")
  Ts = eltype.(data)
  allequal(Ts) || error("All image array layers must have the same element type, got multiple types $(unique(Ts))")
  true
end

function infer_dims_and_format(data, dims, format, layers)
  if isnothing(data)
    isnothing(dims) && error("Image dimensions must be specified when no data is provided.")
    isnothing(format) && error("An image format must be specified when no data is provided.")
  else
    layer = layers == 1 ? data : data[1]
    isnothing(dims) && (dims = collect(size(layer)))
    isnothing(format) && (format = Vk.Format(eltype(layer)))
    isnothing(format) && error("No format could be determined from the data. Please provide an image format.")
  end
  dims, format
end

# # Image views

function ImageView(
  device::Device,
  data = nothing;

  # Image parameters.
  image_flags = nothing,
  image_format = nothing,
  memory_domain = MEMORY_DOMAIN_DEVICE,
  optimal_tiling = true,
  usage_flags = Vk.IMAGE_USAGE_SAMPLED_BIT,
  dims = nothing,
  samples = 1,
  queue_family_indices = queue_family_indices(device),
  sharing_mode = Vk.SHARING_MODE_EXCLUSIVE,
  mip_levels = 1,
  layers = 1,
  layout::Optional{Vk.ImageLayout} = nothing,

  # Image view parameters.
  flags = nothing,
  format = nothing,
  component_mapping = COMPONENT_MAPPING_IDENTITY,
  aspect = nothing,
  layer_range = nothing,
  mip_range = nothing,
  type = nothing,

  submission = isnothing(data) ? nothing : SubmissionInfo(signal_fence = get_fence!(device)))

  isa(format, Type) && (format = Vk.Format(format))
  dims, format = infer_dims_and_format(data, dims, format, layers)
  image_format = something(image_format, format, Some(nothing))
  !isnothing(data) && (usage_flags |= Vk.IMAGE_USAGE_TRANSFER_DST_BIT)
  usage_flags = minimal_image_view_flags(usage_flags)
  image = Image(device; format = image_format, memory_domain, optimal_tiling, usage_flags, dims, samples, queue_family_indices, sharing_mode, layers, mip_levels, flags = image_flags)

  flags = something(flags, Vk.ImageViewCreateFlag())
  aspect = @something(aspect, format == Vk.FORMAT_UNDEFINED ? Vk.IMAGE_ASPECT_COLOR_BIT : aspect_flags(format))
  layer_range = @something(layer_range, Lava.layer_range(image))
  mip_range = @something(mip_range, Lava.mip_range(image))
  type = @something(type, image_view_type(dimensions(image), layer_range))
  view = ImageView(image; format, aspect, layer_range, mip_range, component_mapping, flags)
  !isnothing(data) && copyto!(view, data, device; submission)
  !isnothing(layout) && ensure_layout(device, view, layout)
  view
end

# # Attachments

function Attachment(
  device::Device,
  data = nothing;

  # Image parameters.
  image_flags = nothing,
  image_format = nothing,
  memory_domain = MEMORY_DOMAIN_DEVICE,
  optimal_tiling = true,
  usage_flags = Vk.IMAGE_USAGE_SAMPLED_BIT,
  dims = nothing,
  samples = 1,
  queue_family_indices = queue_family_indices(device),
  sharing_mode = Vk.SHARING_MODE_EXCLUSIVE,
  mip_levels = 1,
  layers = 1,
  layout::Optional{Vk.ImageLayout} = nothing,

  # Image view parameters.
  flags = nothing,
  format = nothing,
  component_mapping = COMPONENT_MAPPING_IDENTITY,
  aspect = nothing,
  layer_range = nothing,
  mip_range = nothing,
  type = nothing,

  # Attachment parameters.
  access::MemoryAccess = READ | WRITE,

  submission = isnothing(data) ? nothing : SubmissionInfo(signal_fence = get_fence!(device)))

  image_format = something(image_format, format, Some(nothing))
  view = ImageView(device, data; image_flags, image_format, memory_domain, optimal_tiling, usage_flags, dims, samples, queue_family_indices, sharing_mode, layers, mip_levels, layout, flags, format, aspect, layer_range, mip_range, type, component_mapping, submission)
  Attachment(view, access)
end

const REQUIRED_IMAGE_VIEW_FLAGS = |(
  Vk.IMAGE_USAGE_SAMPLED_BIT,
  Vk.IMAGE_USAGE_STORAGE_BIT,
  Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
  Vk.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
  Vk.IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
  Vk.IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
  Vk.IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
  Vk.IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT,
  Vk.IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR,
  Vk.IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR,
  Vk.IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR,
  Vk.IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR,
  Vk.IMAGE_USAGE_SAMPLE_WEIGHT_BIT_QCOM,
  Vk.IMAGE_USAGE_SAMPLE_BLOCK_MATCH_BIT_QCOM,
)

function minimal_image_view_flags(usage_flags::Vk.ImageUsageFlag)
  !iszero(usage_flags & REQUIRED_IMAGE_VIEW_FLAGS) && return usage_flags
  usage_flags | Vk.IMAGE_USAGE_SAMPLED_BIT
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
