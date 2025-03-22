function draw_texture(device, vdata, color, depth; prog = texture_program(device), image = nothing, uv_scale = Vec2(0.1, 1.0))
  image = @something(image, read_normal_map(device))
  image_texture = texture_descriptor(Texture(image, setproperties(DEFAULT_SAMPLING, (magnification = Vk.FILTER_LINEAR, minification = Vk.FILTER_LINEAR))))
  invocation_data = @invocation_data prog begin
    b1 = @block vdata
    b2 = @block TextureDrawing(uv_scale, @descriptor image_texture)
    @block TextureData(@address(b1), @address(b2))
  end
  graphics_command(
    DrawIndexed(1:4),
    prog,
    invocation_data,
    RenderTargets(color; depth),
    RenderState(),
    setproperties(ProgramInvocationState(), (;
      primitive_topology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
      triangle_orientation = Vk.FRONT_FACE_COUNTER_CLOCKWISE,
    )),
    @resource_dependencies begin
      @read
      image::Texture
      @write
      (color => (0.08, 0.05, 0.1, 1.0))::Color
      (depth => 0.0)::Depth
    end
  )
end

@testset "Data persistence across cycles" begin
  color = attachment_resource(device, nothing; format = RGBA{Float16}, usage_flags = Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_TRANSFER_DST_BIT | Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT, dims = [1920, 1080])
  depth = attachment_resource(Vk.FORMAT_D32_SFLOAT, dimensions(color))

  @testset "`RenderNode` persistence" begin
    vdata = [
      PosColor(Vec2(-0.7, 0.7), Vec3(1.0, 0.0, 0.0)),
      PosColor(Vec2(0.3, 0.7), Vec3(0.0, 1.0, 0.0)),
      PosColor(Vec2(-0.7, -0.3), Vec3(1.0, 1.0, 1.0)),
      PosColor(Vec2(0.3, -0.3), Vec3(0.0, 0.0, 1.0)),
    ]
    draw = draw_rectangle(device, vdata, color)
    hashes = UInt64[]
    for i in 1:5
      data = render_graphics(device, draw)
      push!(hashes, hash(data))
    end
    @test all(==(0xc92df9461d3cc743), hashes)

    vdata = [
      TextureCoordinates(Vec2(-0.5, -0.5), Vec2(0.0, 0.0)),
      TextureCoordinates(Vec2(-0.5, 0.5), Vec2(0.0, 1.0)),
      TextureCoordinates(Vec2(0.5, -0.5), Vec2(1.0, 0.0)),
      TextureCoordinates(Vec2(0.5, 0.5), Vec2(1.0, 1.0)),
    ]
    draw = draw_texture(device, vdata, color, depth)
    hashes = UInt64[]
    for i in 1:5
      data = render_graphics(device, draw)
      push!(hashes, hash(data))
    end
    @test all(==(0x7c14e3ffe5603da5), hashes)
  end

  @testset "`RenderGraph` persistence" begin
    vdata = [
      TextureCoordinates(Vec2(-0.5, -0.5), Vec2(0.0, 0.0)),
      TextureCoordinates(Vec2(-0.5, 0.5), Vec2(0.0, 1.0)),
      TextureCoordinates(Vec2(0.5, -0.5), Vec2(1.0, 0.0)),
      TextureCoordinates(Vec2(0.5, 0.5), Vec2(1.0, 1.0)),
    ]
    draw = draw_texture(device, vdata, color, depth)
    rg = RenderGraph(device, draw)
    hashes = UInt64[]
    for i in 1:5
      render!(rg)
      data = read_data(device, color)
      finish!(rg)
      push!(hashes, hash(data))
    end
    @test all(==(0x7c14e3ffe5603da5), hashes)

    rg = RenderGraph(device, draw)
    hashes = UInt64[]
    bake!(rg)
    for i in 1:5
      submission = sync_submission(device)
      command_buffer = request_command_buffer(device)
      render(command_buffer, rg)
      execution = Lava.submit!(submission, command_buffer)
      wait(execution)
      data = read_data(device, color)
      push!(hashes, hash(data))
    end
    @test all(==(0x7c14e3ffe5603da5), hashes)
    finish!(rg)
  end
end;
