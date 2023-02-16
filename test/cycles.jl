@testset "Data persistence across cycles" begin
  color = attachment_resource(device, nothing; format = Vk.FORMAT_R16G16B16A16_SFLOAT, usage_flags = Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_TRANSFER_DST_BIT | Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT, dims = [1920, 1080])
  vdata = [
    PosColor(Vec2(-0.5, 0.5), Arr{Float32}(1.0, 0.0, 0.0)),
    PosColor(Vec2(-0.5, -0.5), Arr{Float32}(0.0, 1.0, 0.0)),
    PosColor(Vec2(0.5, 0.5), Arr{Float32}(1.0, 1.0, 1.0)),
    PosColor(Vec2(0.5, -0.5), Arr{Float32}(0.0, 0.0, 1.0)),
  ]
  draw = draw_rectangle(device, vdata, color)
  node = graphics_node([draw])
  hashes = UInt64[]
  for i in 1:5
    data = render_graphics(device, node)
    push!(hashes, hash(data))
  end
  @test all(==(0x9430efd8e0911300), hashes)

  vdata = [
    TextureCoordinates(Vec2(-0.5, 0.5), Vec2(0.0, 0.0)),
    TextureCoordinates(Vec2(-0.5, -0.5), Vec2(0.0, 1.0)),
    TextureCoordinates(Vec2(0.5, 0.5), Vec2(1.0, 0.0)),
    TextureCoordinates(Vec2(0.5, -0.5), Vec2(1.0, 1.0)),
  ]
  draw = draw_texture(device, vdata, color)
  node = graphics_node([draw])
  hashes = UInt64[]
  for i in 1:5
    data = render_graphics(device, node)
    push!(hashes, hash(data))
  end
  @test all(==(0x9eda4cb9b969b269), hashes)
end;
