struct PosColor
  pos::Vec{2,Float32}
  color::Arr{3,Float32}
end

function rectangle_vert(frag_color, position, index, data_address::DeviceAddressBlock)
  data = @load data_address[index + 1U]::PosColor
  (; pos, color) = data
  position[] = Vec(pos.x, pos.y, 0F, 1F)
  frag_color[] = Vec(color[1], color[2], color[3], 1F)
end

function rectangle_frag(out_color, frag_color)
  out_color[] = frag_color
end

function rectangle_program(device)
  vert = @vertex device rectangle_vert(::Vec4::Output, ::Vec4::Output{Position}, ::UInt32::Input{VertexIndex}, ::DeviceAddressBlock::PushConstant)
  frag = @fragment device rectangle_frag(::Vec4::Output, ::Vec4::Input)
  Program(vert, frag)
end

function draw_rectangle(device, vdata, color, prog = rectangle_program(device))
  invocation_data = @invocation_data prog @block vdata
  graphics_command(
    DrawIndexed(1:4),
    prog,
    invocation_data,
    RenderTargets(color),
    RenderState(),
    setproperties(ProgramInvocationState(), (;
      primitive_topology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
      triangle_orientation = Vk.FRONT_FACE_COUNTER_CLOCKWISE,
    )),
    @resource_dependencies begin
      @write
      (color => (0.08, 0.05, 0.1, 1.0))::Color
    end
  )
end

@testset "Rectangle" begin
  vdata = [
    PosColor(Vec2(-0.7, 0.7), Arr{Float32}(1.0, 0.0, 0.0)),
    PosColor(Vec2(0.3, 0.7), Arr{Float32}(0.0, 1.0, 0.0)),
    PosColor(Vec2(-0.7, -0.3), Arr{Float32}(1.0, 1.0, 1.0)),
    PosColor(Vec2(0.3, -0.3), Arr{Float32}(0.0, 0.0, 1.0)),
  ]
  draw = draw_rectangle(device, vdata, color)
  data = render_graphics(device, draw)
  save_test_render("colored_rectangle.png", data, 0xc92df9461d3cc743)
end;
