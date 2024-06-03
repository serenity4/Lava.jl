struct TransferCommand <: CommandImplementation
  src::Resource
  dst::Resource
  blit_filter::Vk.Filter
end
TransferCommand(src, dst; blit_filter = Vk.FILTER_CUBIC_IMG) = TransferCommand(src, dst, blit_filter)

function resource_dependencies(transfer::TransferCommand)
  (; src, dst) = transfer
  Dictionary(
    [src, dst],
    [
      ResourceDependency(RESOURCE_USAGE_TRANSFER_SRC, READ, nothing, samples(src)),
      ResourceDependency(RESOURCE_USAGE_TRANSFER_DST, WRITE, nothing, samples(dst)),
    ]
  )
end

is_blit(transfer::TransferCommand) = !isbuffer(transfer.src) && !isbuffer(transfer.dst) && dimensions(transfer.src) ≠ dimensions(transfer.dst)
is_resolve(transfer::TransferCommand) = !isbuffer(transfer.src) && !isbuffer(transfer.dst) && samples(transfer.src) ≠ samples(transfer.dst)
is_copy(transfer::TransferCommand) = isbuffer(transfer.src) || isbuffer(transfer.dst) || (!is_blit(transfer) && !is_resolve(transfer))

function apply(command_buffer::CommandBuffer, transfer::TransferCommand, resources)
  src = get_physical_resource(resources, transfer.src)
  dst = get_physical_resource(resources, transfer.dst)
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
    samples(src_image) == samples(dst_image) || throw(error("Only transfers between images of identical sample counts are currently supported"))
    if dimensions(src_image) != dimensions(dst_image) || src_image.format != dst_image.format
      # Perform a blit operation instead.
      # TODO: Allow copying for size-compatible image formats instead of blitting,
      # See https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap47.html#formats-compatibility-classes.
      region = Vk.ImageBlit2(C_NULL, Subresource(src), (Vk.Offset3D(src), Vk.Offset3D(dimensions(src)..., 1)), Subresource(dst), (Vk.Offset3D(dst), Vk.Offset3D(dimensions(dst)..., 1)))
      info = Vk.BlitImageInfo2(C_NULL, src_image, image_layout(src), dst_image, image_layout(dst), [region], Vk.FILTER_LINEAR)
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
