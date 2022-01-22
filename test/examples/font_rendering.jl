function program_3(device, positions, ppm)
    prog = Program(device, ShaderSpecification(shader_file("glyph.vert"), GLSL), ShaderSpecification(shader_file("glyph.frag"), GLSL))

    fg = FrameGraph(device)
    add_color_attachment(fg)

    font = OpenTypeFont(font_file("juliamono-regular.ttf"))
    curves = map(OpenType.curves(font['C'])) do curve
        map(curve) do points
            map(Float32, points)
        end
    end
    curve_buffer = buffer(device, curves)
    vdata = [(pos..., UInt32(0), UInt32(length(curves))) for pos in positions]
    register(fg.frame, :curve_buffer, curve_buffer)

    add_pass!(fg, :main, RenderPass((0,0,1920, 1080))) do rec
        set_program(rec, prog)
        ds = draw_state(rec)
        set_draw_state(rec, @set ds.program_state.primitive_topology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP)
        set_material(rec,
            RGBA{Float32}(1., 1., 0., 1.), # text color
            ppm, # pixel per em
            device_address(curve_buffer);
            alignment = 8
        )
        draw(rec, TargetAttachments([:color]), vdata, collect(1:4); alignment = 8)
    end

    usage = @resource_usages begin
        color::Color = main()
    end
    add_resource_usage!(fg, usage)
    clear_attachments(fg, :main, [:color => (0.08, 0.05, 0.1, 1.)])
    fg
end

@testset "Font rendering" begin
    positions = [
        (-0.8f0, -0.8f0),
        (0.8f0, -0.8f0),
        (-0.8f0, 0.8f0),
        (0.8f0, 0.8f0),
    ]
    fg = program_3(device, positions, 12f0)
    @test wait(render(fg))
    # FIXME: debug why font shader fails
    # data = collect(RGBA{Float16}, image(fg.frame.resources[:color].data), device)
    # save_test_render("glyph_A.png", data, hash(data))
end
