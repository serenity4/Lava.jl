@enum AttachmentUsage begin
    READ_ONLY
    WRITE_ONLY
    READ_WRITE
end

struct Attachment{IV<:ImageView}
    view::IV
    usage::AttachmentUsage
end

@forward Attachment.view dims, format, samples

function load_op(usage::AttachmentUsage)
    @match usage begin
        &READ_ONLY => Vk.ATTACHMENT_LOAD_OP_DONT_CARE
        &WRITE_ONLY || &READ_WRITE => Vk.ATTACHMENT_LOAD_OP_CLEAR
    end
end

function store_op(usage::AttachmentUsage)
    @match usage begin
        &READ_ONLY => Vk.ATTACHMENT_STORE_OP_DONT_CARE
        &WRITE_ONLY || &READ_WRITE => Vk.ATTACHMENT_STORE_OP_STORE
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
