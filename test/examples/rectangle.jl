struct VertexDataRectangle
    pos::Vec{2,Float32}
    color::Arr{3,Float32}
end

function rectangle_vert(frag_color, position, index, dd)
    vd = Pointer{Vector{VertexDataRectangle}}(dd.vertex_data)[index]
    (; pos, color) = vd
    position[] = Vec(pos.x, pos.y, 0F, 1F)
    frag_color[] = Vec(color[0U], color[1U], color[2U], 1F)
end

function rectangle_frag(out_color, frag_color)
    out_color[] = frag_color
end

function program_1(device, vdata)
    vert_interface = ShaderInterface(
        storage_classes = [SPIRV.StorageClassOutput, SPIRV.StorageClassOutput, SPIRV.StorageClassInput, SPIRV.StorageClassPushConstant],
        variable_decorations = dictionary([
            1 => dictionary([SPIRV.DecorationLocation => [0U]]),
            2 => dictionary([SPIRV.DecorationBuiltIn => [SPIRV.BuiltInPosition]]),
            3 => dictionary([SPIRV.DecorationBuiltIn => [SPIRV.BuiltInVertexIndex]]),
        ]),
        features = SPIRV_FEATURES,
    )

    frag_interface = ShaderInterface(
        execution_model = SPIRV.ExecutionModelFragment,
        storage_classes = [SPIRV.StorageClassOutput, SPIRV.StorageClassInput],
        variable_decorations = dictionary([
            1 => dictionary([SPIRV.DecorationLocation => [0U]]),
            2 => dictionary([SPIRV.DecorationLocation => [0U]]),
        ]),
        features = SPIRV_FEATURES,
    )

    vert_shader = @shader vert_interface rectangle_vert(::Vec{4, Float32}, ::Vec{4, Float32}, ::UInt32, ::DrawData)
    frag_shader = @shader frag_interface rectangle_frag(::Vec{4, Float32}, ::Vec{4, Float32})
    prog = Program(device, vert_shader, frag_shader)

    fg = FrameGraph(device)
    add_color_attachment(fg)
    add_pass!(fg, :main, RenderPass((0,0,1920,1080))) do rec
        set_program(rec, prog)
        ds = draw_state(rec)
        set_draw_state(rec, @set ds.program_state.primitive_topology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP)
        draw(rec, TargetAttachments([:color]), vdata, collect(1:4); alignment = 4)
    end

    usage = @resource_usages begin
        color::Color = main()
    end
    add_resource_usage!(fg, usage)
    clear_attachments(fg, :main, [:color => (0.08, 0.05, 0.1, 1.)])
    fg
end

@testset "Rectangle" begin
    vdata = [
        (-0.5f0, -0.5f0, RGB{Float32}(1., 0., 0.)),
        (0.5f0, -0.5f0, RGB{Float32}(1., 1., 1.)),
        (-0.5f0, 0.5f0, RGB{Float32}(0., 1., 0.)),
        (0.5f0, 0.5f0, RGB{Float32}(0., 0., 1.)),
    ]
    fg = program_1(device, vdata)
    snoop = Lava.SnoopCommandBuffer()
    render(fg; command_buffer = snoop, submit = false)
    @test length(snoop.records) == 8
    @test fg.frame.gd.index_list == [1, 2, 3, 4]
    ib = collect(UInt32, fg.frame.gd.index_buffer[])
    @test ib == UInt[0, 1, 2, 3] && sizeof(ib) == 16
    @test fg.frame.gd.allocator.last_offset == 20 * 4

    fg = program_1(device, vdata)
    @test wait(render(fg))
    data = collect(RGBA{Float16}, image(fg.frame.resources[:color].data), device)
    save_test_render("colored_rectangle.png", data, 0x9430efd8e0911300)
end
