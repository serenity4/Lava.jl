function program_2(device, vdata)
    prog = Program(device, ShaderSpecification(shader_file("texture.vert"), GLSL), ShaderSpecification(shader_file("texture.frag"), GLSL))

    fg = FrameGraph(device)
    add_color_attachment(fg)

    normal = load(texture_file("normal.png"))
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

@testset "Texture drawing" begin
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
