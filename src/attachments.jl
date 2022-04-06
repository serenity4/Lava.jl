Vk.@bitmask_flag MemoryAccess::UInt32 begin
  READ = 1
  WRITE = 2
  NO_ACCESS = 4
end

struct Attachment{IV<:ImageView}
  view::IV
  usage::MemoryAccess
end

@forward Attachment.view (dims, format, samples, image_layout, image, aspect, layer_range, mip_range)

function Base.similar(att::Attachment; memory_domain = nothing, image_usage = image(att).usage, attachment_usage = att.usage, is_linear = image(att).is_linear)
  img = similar(image(att); memory_domain, usage = image_usage, is_linear)
  Attachment(View(img), attachment_usage)
end

function load_op(usage::MemoryAccess, clear::Bool)
  clear && return Vk.ATTACHMENT_LOAD_OP_CLEAR
  (READ in usage || (WRITE in usage && !clear)) && return Vk.ATTACHMENT_LOAD_OP_LOAD
  Vk.ATTACHMENT_LOAD_OP_DONT_CARE
end

function store_op(usage::MemoryAccess)
  WRITE in usage && return Vk.ATTACHMENT_STORE_OP_STORE
  Vk.ATTACHMENT_STORE_OP_DONT_CARE
end
