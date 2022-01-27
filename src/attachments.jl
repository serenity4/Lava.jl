Vk.@bitmask_flag MemoryAccess::UInt32 begin
  READ = 1
  WRITE = 2
  NO_ACCESS = 4
end

struct Attachment{IV<:ImageView}
  view::IV
  usage::MemoryAccess
end

@forward Attachment.view dims, format, samples, image_layout, image

function load_op(usage::MemoryAccess, clear::Bool)
  @match usage begin
    &READ => Vk.ATTACHMENT_LOAD_OP_LOAD
    &WRITE || &(READ | WRITE) => clear ? Vk.ATTACHMENT_LOAD_OP_CLEAR : Vk.ATTACHMENT_LOAD_OP_DONT_CARE
    _ => Vk.ATTACHMENT_LOAD_OP_DONT_CARE
  end
end

function store_op(usage::MemoryAccess)
  @match usage begin
    &WRITE || &(READ | WRITE) => Vk.ATTACHMENT_STORE_OP_STORE
    &READ => Vk.ATTACHMENT_STORE_OP_DONT_CARE
    _ => nothing
  end
end

function Vk.AttachmentDescription2(
  att::Attachment,
  clear::Bool,
  initial_layout::Vk.ImageLayout,
  final_layout::Vk.ImageLayout,
  aspect::Vk.ImageAspectFlag,
)
  _load_op, _store_op = if Vk.IMAGE_ASPECT_COLOR_BIT in aspect
    (load_op(att.usage, clear), store_op(att.usage))
  else
    (Vk.ATTACHMENT_LOAD_OP_DONT_CARE, Vk.ATTACHMENT_STORE_OP_DONT_CARE)
  end

  stencil_load_op, stencil_store_op = if Vk.IMAGE_ASPECT_STENCIL_BIT in aspect
    (load_op(att.usage, clear), store_op(att.usage))
  else
    (Vk.ATTACHMENT_LOAD_OP_DONT_CARE, Vk.ATTACHMENT_STORE_OP_DONT_CARE)
  end

  Vk.AttachmentDescription2(
    format(att),
    samples(att),
    _load_op,
    _store_op,
    stencil_load_op,
    stencil_store_op,
    initial_layout,
    final_layout,
  )
end

@auto_hash_equals struct TargetAttachments
  color::Vector{Symbol}
  depth::Vector{Symbol}
  stencil::Vector{Symbol}
end

TargetAttachments(color; depth = [], stencil = []) = TargetAttachments(color, depth, stencil)
