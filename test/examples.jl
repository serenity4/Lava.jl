function color_attachment(device)
    usage = Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT | Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_TRANSFER_DST_BIT
    color_image = allocate!(ImageBlock(device, (1920, 1080), Vk.FORMAT_R16G16B16A16_SFLOAT, usage), MEMORY_DOMAIN_DEVICE)
    color_attachment = Attachment(View(color_image), WRITE)
end

function add_color_attachment(fg::FrameGraph)
    color_attachm = color_attachment(fg.device)
    register(fg.frame, :color, color_attachm)
    add_resource!(fg, :color, AttachmentResourceInfo(Lava.format(color_attachm)))
end

function save_test_render(filename, data, h::UInt; tmp = false)
    filename = render_file(filename; tmp)
    ispath(filename) && rm(filename)
    save(filename, data')
    @test stat(filename).size > 0
    @test hash(data) == h
end

include("examples/rectangle.jl")
