@testset "Building a frame graph" begin
    frame = Frame(device)
    fg = FrameGraph(device, frame)

    add_pass!(identity, fg, :gbuffer, RenderPass((0,0,1920,1080)))
    add_pass!(identity, fg, :lighting, RenderPass((0,0,1920,1080)))
    add_pass!(identity, fg, :adapt_luminance, RenderPass((0,0,1920,1080)))
    add_pass!(identity, fg, :combine, RenderPass((0,0,1920,1080)))
    # can't add a pass more than once
    @test_throws ErrorException add_pass!(identity, fg, :combine, RenderPass((0,0,1920,1080)))

    add_resource!(fg, :vbuffer, BufferResourceInfo(1024))
    add_resource!(fg, :ibuffer, BufferResourceInfo(1024))
    add_resource!(fg, :average_luminance, ImageResourceInfo(Vk.FORMAT_R32G32B32A32_SFLOAT))
    add_resource!(fg, :emissive, AttachmentResourceInfo(Vk.FORMAT_R32G32B32A32_SFLOAT))
    add_resource!(fg, :albedo, AttachmentResourceInfo(Vk.FORMAT_R32G32B32A32_SFLOAT))
    add_resource!(fg, :normal, AttachmentResourceInfo(Vk.FORMAT_R32G32B32A32_SFLOAT))
    add_resource!(fg, :pbr, AttachmentResourceInfo(Vk.FORMAT_R32G32B32A32_SFLOAT))
    add_resource!(fg, :color, AttachmentResourceInfo(Vk.FORMAT_R32G32B32A32_SFLOAT))
    add_resource!(fg, :output, AttachmentResourceInfo(Vk.FORMAT_R32G32B32A32_SFLOAT))
    add_resource!(fg, :depth, AttachmentResourceInfo(Vk.FORMAT_D32_SFLOAT))
    # can't add a resource more than once
    @test_throws ErrorException add_resource!(fg, :depth, AttachmentResourceInfo(Vk.FORMAT_D32_SFLOAT))

    # imported
    add_resource!(fg, :shadow_main, AttachmentResourceInfo(Vk.FORMAT_D32_SFLOAT))
    add_resource!(fg, :shadow_near, AttachmentResourceInfo(Vk.FORMAT_D32_SFLOAT))
    add_resource!(fg, :bloom_downsample_3, AttachmentResourceInfo(Vk.FORMAT_R32G32B32A32_SFLOAT))

    usages = @resource_usages begin
        emissive::Color, albedo::Color, normal::Color, pbr::Color, depth::Depth = gbuffer(vbuffer::Buffer::Vertex, ibuffer::Buffer::Index)
        color::Color = lighting(emissive::Color, albedo::Color, normal::Color, pbr::Color, depth::Depth, shadow_main::Texture, shadow_near::Texture)
        average_luminance::Image::Storage = adapt_luminance(average_luminance::Image::Storage, bloom_downsample_3::Texture)
        output::Color = combine(color::Color, average_luminance::Texture)
    end

    add_resource_usage!(fg, usages)
    Lava.resolve_attributes!(fg)

    @test Lava.buffer_usage(fg, :vbuffer) == Vk.BUFFER_USAGE_VERTEX_BUFFER_BIT
    @test Lava.image_usage(fg, :depth) == Vk.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
    @test Lava.image_usage(fg, :shadow_main) == Vk.IMAGE_USAGE_SAMPLED_BIT

    for resource in keys(fg.resources)
        @test Int(Lava.resource_attribute(fg, resource, :usage)) ≠ 0
    end
end

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

function program_1(device, vdata)
    prog = Program(device, ShaderSpecification(resource("dummy.vert"), GLSL), ShaderSpecification(resource("dummy.frag"), GLSL))

    fg = FrameGraph(device)
    add_color_attachment(fg)
    add_pass!(fg, :main, RenderPass((0,0,1920,1080))) do rec
        set_program(rec, prog)
        ds = draw_state(rec)
        set_draw_state(rec, @set ds.program_state.primitive_topology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP)
        draw(rec, TargetAttachments([:color]), vdata, collect(1:4))
    end

    usage = @resource_usages begin
        color::Color = main()
    end
    add_resource_usage!(fg, usage)
    clear_attachments(fg, :main, [:color => (0.08, 0.05, 0.1, 1.)])
    fg
end

function program_2(device, vdata)
    prog = Program(device, ShaderSpecification(resource("texture.vert"), GLSL), ShaderSpecification(resource("texture.frag"), GLSL))

    fg = FrameGraph(device)
    add_color_attachment(fg)

    normal = load(joinpath(@__DIR__, "resources", "normal.png"))
    normal = convert(Matrix{RGBA{Float16}}, normal)
    normal_map = image(device, normal, Vk.FORMAT_R16G16B16A16_SFLOAT; usage = Vk.IMAGE_USAGE_SAMPLED_BIT)
    register(fg.frame, :normal_map, normal_map)
    add_resource!(fg, :normal_map, ImageResourceInfo(Lava.format(normal_map)))

    add_pass!(fg, :main, RenderPass((0,0,1920, 1080))) do rec
        set_program(rec, prog)
        ds = draw_state(rec)
        set_draw_state(rec, @set ds.program_state.primitive_topology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP)
        set_material(rec,
            (0.1f0, 1f0), # uv scaling coefficients
            Texture(:normal_map, setproperties(DEFAULT_SAMPLING, (magnification = Vk.FILTER_LINEAR, minification = Vk.FILTER_LINEAR))),
        )
        draw(rec, TargetAttachments([:color]), vdata, collect(1:4))
    end

    usage = @resource_usages begin
        color::Color = main(normal_map::Texture)
    end
    add_resource_usage!(fg, usage)
    clear_attachments(fg, :main, [:color => (0.08, 0.05, 0.1, 1.)])
    fg
end

function save_test_render(filename, data, h::UInt)
    filename = joinpath(@__DIR__, filename)
    ispath(filename) && rm(filename)
    save(filename, data')
    @test stat(filename).size > 0
    @test hash(data) == h
end

@testset "Rendering" begin
    vdata = [
        (-0.5f0, -0.5f0, 1.0, RGB{Float32}(1., 0., 0.), 1f0),
        (0.5f0, -0.5f0, 1.0, RGB{Float32}(1., 1., 1.), 1f0),
        (-0.5f0, 0.5f0, 1.0, RGB{Float32}(0., 1., 0.), 1f0),
        (0.5f0, 0.5f0, 1.0, RGB{Float32}(0., 0., 1.), 1f0),
    ]
    fg = program_1(device, vdata)
    snoop = Lava.SnoopCommandBuffer()
    render(fg; command_buffer = snoop, submit = false)
    @test length(snoop.records) == 8
    @test fg.frame.gd.index_list == [1, 2, 3, 4]
    ib = collect(UInt32, fg.frame.gd.index_buffer[])
    @test ib == UInt[0, 1, 2, 3] && sizeof(ib) == 16
    @test fg.frame.gd.allocator.last_offset == sizeof(vdata)
    vd_raw = collect(memory(fg.frame.gd.allocator.buffer), 80)

    fg = program_1(device, vdata)
    @test wait(render(fg))
    data = collect(RGBA{Float16}, image(fg.frame.resources[:color].data), device)
    save_test_render("colored_rectangle.png", data, 0x9430efd8e0911300)

    vdata = [
        (-0.5f0, -0.5f0, 0f0, 0f0),
        (0.5f0, -0.5f0, 1f0, 0f0),
        (-0.5f0, 0.5f0, 0f0, 1f0),
        (0.5f0, 0.5f0, 1f0, 1f0),
    ]
    fg = program_2(device, vdata)
    @test wait(render(fg))
    data = collect(RGBA{Float16}, image(fg.frame.resources[:color].data), device)
    save_test_render("distorted_normal_map.png", data, 0x9eda4cb9b969b269)
end
