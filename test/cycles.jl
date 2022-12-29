function invocation_cycles_rectangle(device, vdata, color, prog = rectangle_program(device))
  invocation_data = @invocation_data begin
    @block vdata
  end
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

function invocation_cycles_texture(device, vdata, color, normal_map, prog = texture_program(device))
  normal_map_texture = texture_descriptor(Texture(normal_map, setproperties(DEFAULT_SAMPLING, (magnification = Vk.FILTER_LINEAR, minification = Vk.FILTER_LINEAR))))
  invocation_data = @invocation_data begin
    b1 = @block vdata
    b2 = @block TextureDrawing(Vec2(0.1, 1.0), @descriptor normal_map_texture)
    @block TextureData(@address(b1), @address(b2))
  end
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
      @read
      normal_map::Texture
      @write
      (color => (0.08, 0.05, 0.1, 1.0))::Color
    end
  )
end

@testset "Data persistence across cycles" begin
  color = attachment_resource(device, nothing; format = Vk.FORMAT_R16G16B16A16_SFLOAT, usage_flags = Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_TRANSFER_DST_BIT | Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT, dims = [1920, 1080])
  vdata = [
    PosColor(Vec2(-0.5, 0.5), Arr{Float32}(1.0, 0.0, 0.0)),
    PosColor(Vec2(-0.5, -0.5), Arr{Float32}(0.0, 1.0, 0.0)),
    PosColor(Vec2(0.5, 0.5), Arr{Float32}(1.0, 1.0, 1.0)),
    PosColor(Vec2(0.5, -0.5), Arr{Float32}(0.0, 0.0, 1.0)),
  ]
  invocation = invocation_cycles_rectangle(device, vdata, color)
  graphics = graphics_node(invocation)

  hashes = UInt64[]
  for i in 1:5
    data = render_graphics(device, graphics)
    push!(hashes, hash(data))
  end
  @test all(==(0x9430efd8e0911300), hashes)

  vdata = [
    TextureCoordinates(Vec2(-0.5, 0.5), Vec2(0.0, 0.0)),
    TextureCoordinates(Vec2(-0.5, -0.5), Vec2(0.0, 1.0)),
    TextureCoordinates(Vec2(0.5, 0.5), Vec2(1.0, 0.0)),
    TextureCoordinates(Vec2(0.5, -0.5), Vec2(1.0, 1.0)),
  ]
  hashes = UInt64[]
  normal_map = read_normal_map(device)
  invocation = invocation_cycles_texture(device, vdata, color, normal_map)
  graphics = graphics_node(invocation)
  for i in 1:5
    data = render_graphics(device, graphics)
    push!(hashes, hash(data))
  end
  @test all(==(0x9eda4cb9b969b269), hashes)
end;
