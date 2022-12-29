struct PosColor
  pos::Vec{2,Float32}
  color::Arr{3,Float32}
end

function rectangle_vert(frag_color, position, index, data_address::DeviceAddressBlock)
  data = @load data_address[index]::PosColor
  (; pos, color) = data
  position[] = Vec(pos.x, pos.y, 0F, 1F)
  frag_color[] = Vec(color[0U], color[1U], color[2U], 1F)
end

function rectangle_frag(out_color, frag_color)
  out_color[] = frag_color
end

function rectangle_program(device)
  vert = @vertex device.spirv_features rectangle_vert(::Output::Vec4, ::Output{Position}::Vec4, ::Input{VertexIndex}::UInt32, ::PushConstant::DeviceAddressBlock)
  frag = @fragment device.spirv_features rectangle_frag(::Output::Vec4, ::Input::Vec4)
  Program(device, vert, frag)
end

function rectangle_invocation(device, vdata, color, prog = rectangle_program(device))
  invocation_data = @invocation_data @block vdata
  ProgramInvocation(
    prog,
    DrawIndexed(1:4),
    RenderTargets(color),
    invocation_data,
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
    PosColor(Vec2(-0.5, 0.5), Arr{Float32}(1.0, 0.0, 0.0)),
    PosColor(Vec2(-0.5, -0.5), Arr{Float32}(0.0, 1.0, 0.0)),
    PosColor(Vec2(0.5, 0.5), Arr{Float32}(1.0, 1.0, 1.0)),
    PosColor(Vec2(0.5, -0.5), Arr{Float32}(0.0, 0.0, 1.0)),
  ]
  invocation = rectangle_invocation(device, vdata, color)
  data = render_graphics(device, graphics_node(invocation))
  save_test_render("colored_rectangle.png", data, 0x9430efd8e0911300)
end;
