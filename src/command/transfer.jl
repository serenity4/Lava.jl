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

is_blit(transfer::TransferCommand) = !isbuffer(transfer.src) && !isbuffer(transfer.dst) && dims(transfer.src.data) ≠ dims(transfer.dst.data)
is_resolve(transfer::TransferCommand) = !isbuffer(transfer.src) && !isbuffer(transfer.dst) && samples(transfer.src.data) ≠ samples(transfer.dst.data)
is_copy(transfer::TransferCommand) = isbuffer(transfer.src) || isbuffer(transfer.dst) || (!is_blit(transfer) && !is_resolve(transfer))

function apply(cb::CommandBuffer, transfer::TransferCommand, resources)
  src = get_physical_resource(resources, transfer.src)
  dst = get_physical_resource(resources, transfer.dst)
  if isbuffer(src) && isbuffer(dst)
    src, dst = src.buffer, dst.buffer
    info = Vk.BufferCopy(src.offset, dst.offset, src.size)
    Vk.cmd_copy_buffer(cb, src, dst, [info])
  elseif (isimage(src) || isattachment(src)) && (isimage(dst) || isattachment(dst))
    src = isimage(src) ? src.image : src.attachment
    dst = isimage(dst) ? dst.image : dst.attachment
    src_image = get_image(src)
    dst_image = get_image(dst)
    # TODO: Implement resolve operations for multisampled images.
    # We could consider the resolve attachment to be a "dynamic" resource dependency, that would be added to `resource_dependencies` above.
    samples(src_image) == samples(dst_image) || throw(error("Only transfers between images of identical sample counts are currently supported"))
    if dims(src_image) != dims(dst_image)
      # Perform a blit operation instead.
      info = Vk.BlitImageInfo2(src_image, image_layout(src), dst_image, image_layout(dst), [], Vk.FILTER_CUBIC_IMG)
      Vk.cmd_blit_image_2(cb, info)
    else
      info = Vk.ImageCopy(subresource_layers(src), Vk.Offset3D(src), subresource_layers(dst), Vk.Offset3D(dst), Vk.Extent3D(src))
      Vk.cmd_copy_image(cb,
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
      dims(dst)...,
      subresource_layers(dst),
      Vk.Offset3D(dst),
      Vk.Extent3D(dst),
    )
    Vk.cmd_copy_buffer_to_image(cb, src, dst_image, Vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, [info])
  elseif isbuffer(dst)
    src = isimage(src) ? src.image : src.attachment
    dst = dst.buffer
    src_image = get_image(src)
    info = Vk.BufferImageCopy(
      dst.offset,
      dims(src)...,
      subresource_layers(src),
      Vk.Offset3D(src),
      Vk.Extent3D(src),
    )
    Vk.cmd_copy_image_to_buffer(cb, src_image, Vk.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, [info])
  end
end
