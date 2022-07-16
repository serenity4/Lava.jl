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

function rectangle_program(device)
  vert = @vertex device.spirv_features rectangle_vert(::Output::Vec{4,Float32}, ::Output{Position}::Vec{4,Float32}, ::Input{VertexIndex}::UInt32, ::PushConstant::DrawData)
  frag = @fragment device.spirv_features rectangle_frag(::Output::Vec{4,Float32}, ::Input::Vec{4,Float32})
  Program(device, vert, frag)
end

function program_1(device, vdata, color)
  rg = RenderGraph(device)

  graphics = RenderNode(render_area = RenderArea(1920, 1080), stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT | Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)

  @add_resource_dependencies rg begin
    (color => (0.08, 0.05, 0.1, 1.0))::Color = graphics()
  end

  rec = StatefulRecording()
  set_program(rec, rectangle_program(device))
  set_invocation_state(rec, setproperties(invocation_state(rec), (;
    primitive_topology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
    triangle_orientation = Vk.FRONT_FACE_COUNTER_CLOCKWISE,
  )))
  draw(graphics, rec, rg, vdata, collect(1:4), color)

  rg
end

@testset "Rectangle" begin
  vdata = [
    VertexDataRectangle(Vec2(-0.5, 0.5), Arr{Float32}(1.0, 0.0, 0.0)),
    VertexDataRectangle(Vec2(-0.5, -0.5), Arr{Float32}(0.0, 1.0, 0.0)),
    VertexDataRectangle(Vec2(0.5, 0.5), Arr{Float32}(1.0, 1.0, 1.0)),
    VertexDataRectangle(Vec2(0.5, -0.5), Arr{Float32}(0.0, 0.0, 1.0)),
  ]
  rg = program_1(device, vdata, pcolor)

  render(rg)
  data = collect(RGBA{Float16}, color.view.image, device)
  save_test_render("colored_rectangle.png", data, 0x9430efd8e0911300)
end;
