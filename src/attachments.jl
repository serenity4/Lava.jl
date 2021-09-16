Vk.@bitmask_flag MemoryAccess::UInt32 begin
    NO_ACCESS = 0
    READ = 1
    WRITE = 2
end

struct Attachment{IV<:ImageView}
    view::IV
    usage::MemoryAccess
end

@forward Attachment.view dims, format, samples

function load_op(usage::MemoryAccess)
    @match usage begin
        &READ => Vk.ATTACHMENT_LOAD_OP_DONT_CARE
        &READ || &(READ | WRITE) => Vk.ATTACHMENT_LOAD_OP_CLEAR
        _ => nothing
    end
end

function store_op(usage::MemoryAccess)
    @match usage begin
        &READ => Vk.ATTACHMENT_STORE_OP_DONT_CARE
        &READ || &(READ | WRITE) => Vk.ATTACHMENT_STORE_OP_STORE
        _ => nothing
    end
end

function Vk.AttachmentDescription(att::Attachment)
    asp = aspect(att)

    _load_op, _store_op = if Vk.IMAGE_ASPECT_COLOR_BIT in asp
        (load_op(att.usage), load_op(att.usage))
    else
        (Vk.ATTACHMENT_LOAD_OP_DONT_CARE, Vk.ATTACHMENT_STORE_OP_DONT_CARE)
    end

    stencil_load_op, stencil_store_op = if Vk.IMAGE_ASPECT_STENCIL_BIT in asp
        (load_op(att.usage), load_op(att.usage))
    else
        (Vk.ATTACHMENT_LOAD_OP_DONT_CARE, Vk.ATTACHMENT_STORE_OP_DONT_CARE)
    end

    Vk.AttachmentDescription(
        format(att),
        samples(att),
        _load_op,
        _store_op,
        stencil_load_op,
        stencil_store_op,
        # missing layouts
    )
end
