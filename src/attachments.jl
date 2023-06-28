@bitmask MemoryAccess::UInt32 begin
  NO_ACCESS = 0
  READ = 1
  WRITE = 2
end

struct Attachment
  view::ImageView
  access::MemoryAccess
end

is_multisampled(att::Attachment) = is_multisampled(att.view.image)

@forward_methods Attachment field = :view aspect_flags image_layout samples dimensions get_image subresource_layers Vk.Offset3D Vk.Extent3D

function Base.similar(att::Attachment; memory_domain = nothing, usage_flags = att.view.image.usage_flags, access = att.access, is_linear = att.view.image.is_linear)
  (; view) = att
  img = similar(view.image; memory_domain, usage_flags, is_linear)
  Attachment(ImageView(img; view.format, view.aspect, view.mip_range, view.layer_range), access)
end

function load_op(access::MemoryAccess, clear::Bool)
  clear && return Vk.ATTACHMENT_LOAD_OP_CLEAR
  (READ in access || (WRITE in access && !clear)) && return Vk.ATTACHMENT_LOAD_OP_LOAD
  Vk.ATTACHMENT_LOAD_OP_DONT_CARE
end

function store_op(access::MemoryAccess)
  WRITE in access && return Vk.ATTACHMENT_STORE_OP_STORE
  Vk.ATTACHMENT_STORE_OP_DONT_CARE
end
