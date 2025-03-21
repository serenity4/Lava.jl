struct TransferCommand <: CommandImplementation
  src::Resource
  dst::Resource
  blit_filter::Vk.Filter
  multisample_resolve::Optional{Resource}
end
TransferCommand(src, dst; blit_filter = Vk.FILTER_LINEAR) = TransferCommand(src, dst, blit_filter, multisample_resolve(src, dst))

function multisample_resolve(src, dst)
  (isbuffer(src) || isbuffer(dst)) && return nothing
  (; name) = src
  src = isimage(src) ? src.data::Union{Image,LogicalImage} : src.data::Union{Attachment,LogicalAttachment}
  dst = isimage(dst) ? dst.data::Union{Image,LogicalImage} : dst.data::Union{Attachment,LogicalAttachment}
  samples(src) == samples(dst) && return nothing
  samples(src) > 1 || error("Image transfers require a destination image with the same number of samples, or will perform a multisampling resolve operation requiring a single-sampled destination image.")
  samples(dst) == 1 || error("Multisampling resolution requires a single-sampled destination image.")
  src_image = get_image(src)
  name = isnothing(name) ? nothing : Symbol(name, :_resolve_transient)
  image_resource(src_image.format, src_image.dims; src_image.layers, src_image.mip_levels, samples = 1, name)
end

function resource_dependencies(transfer::TransferCommand)
  (; src, dst, multisample_resolve) = transfer
  dependencies = Dictionary(
    [src, dst],
    [
      ResourceDependency(RESOURCE_USAGE_TRANSFER_SRC, READ, nothing, samples(src)),
      ResourceDependency(RESOURCE_USAGE_TRANSFER_DST, WRITE, nothing, samples(dst)),
    ]
  )

  !isnothing(multisample_resolve) && insert!(dependencies, multisample_resolve, ResourceDependency(RESOURCE_USAGE_TRANSFER_SRC | RESOURCE_USAGE_TRANSFER_DST, READ | WRITE, nothing, 1))

  dependencies
end

is_blit(transfer::TransferCommand) = !isbuffer(transfer.src) && !isbuffer(transfer.dst) && (dimensions(transfer.src) ≠ dimensions(transfer.dst) || image_format(transfer.src) ≠ image_format(transfer.dst))
is_resolve(transfer::TransferCommand) = !isbuffer(transfer.src) && !isbuffer(transfer.dst) && samples(transfer.src) ≠ samples(transfer.dst)
is_copy(transfer::TransferCommand) = isbuffer(transfer.src) || isbuffer(transfer.dst) || (!is_blit(transfer) && !is_resolve(transfer))

function apply(command_buffer::CommandBuffer, transfer::TransferCommand, materialized_resources)
  src = get_physical_resource(materialized_resources, transfer.src)
  dst = get_physical_resource(materialized_resources, transfer.dst)

  if !isnothing(transfer.multisample_resolve)
    # Perform a multisampling resolution step, then transfer again.
    multisample_resolve = get_physical_resource(materialized_resources, transfer.multisample_resolve)
    src = isimage(src) ? src.image : src.attachment
    aux = multisample_resolve.image
    regions = [Vk.ImageResolve(Subresource(src), Vk.Offset3D(src), Subresource(aux), Vk.Offset3D(aux), Vk.Extent3D(src))]
    Vk.cmd_resolve_image(command_buffer, get_image(src), image_layout(src), get_image(aux), image_layout(aux), regions)
    return apply(command_buffer, TransferCommand(multisample_resolve, transfer.dst, transfer.blit_filter, nothing), materialized_resources)
  end

  if isbuffer(src) && isbuffer(dst)
    src, dst = src.buffer, dst.buffer
    info = Vk.BufferCopy(src.offset, dst.offset, src.size)
    Vk.cmd_copy_buffer(command_buffer, src, dst, [info])
  elseif (isimage(src) || isattachment(src)) && (isimage(dst) || isattachment(dst))
    src = isimage(src) ? src.image : src.attachment
    dst = isimage(dst) ? dst.image : dst.attachment
    src_image = get_image(src)
    dst_image = get_image(dst)
    # TODO: Implement resolve operations for multisampled images.
    # We could consider the resolve attachment to be a "dynamic" resource dependency, that would be added to `resource_dependencies` above.
    if dimensions(src_image) != dimensions(dst_image) || src_image.format != dst_image.format
      # Perform a blit operation instead.
      # TODO: Allow copying for size-compatible image formats instead of blitting,
      # See https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap47.html#formats-compatibility-classes.
      region = Vk.ImageBlit2(C_NULL, Subresource(src), (Vk.Offset3D(src), Vk.Offset3D(dimensions(src)..., 1)), Subresource(dst), (Vk.Offset3D(dst), Vk.Offset3D(dimensions(dst)..., 1)))
      info = Vk.BlitImageInfo2(C_NULL, src_image, image_layout(src), dst_image, image_layout(dst), [region], transfer.blit_filter)
      Vk.cmd_blit_image_2(command_buffer, info)
    else
      info = Vk.ImageCopy(Subresource(src), Vk.Offset3D(src), Subresource(dst), Vk.Offset3D(dst), Vk.Extent3D(src))
      Vk.cmd_copy_image(command_buffer,
        src_image, image_layout(src),
        dst_image, image_layout(dst),
        [info],
      )
    end
  elseif isbuffer(src)
    src = src.buffer
    dst = isimage(dst) ? dst.image : dst.attachment
    dst_image = get_image(dst)
    info = Vk.BufferImageCopy(
      src.offset,
      dimensions(dst)...,
      Subresource(dst),
      Vk.Offset3D(dst),
      Vk.Extent3D(dst),
    )
    Vk.cmd_copy_buffer_to_image(command_buffer, src, dst_image, Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, [info])
  elseif isbuffer(dst)
    src = isimage(src) ? src.image : src.attachment
    dst = dst.buffer
    src_image = get_image(src)
    info = Vk.BufferImageCopy(
      dst.offset,
      dimensions(src)...,
      Subresource(src),
      Vk.Offset3D(src),
      Vk.Extent3D(src),
    )
    Vk.cmd_copy_image_to_buffer(command_buffer, src_image, Vk.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, [info])
  end
end
