@testset "Data persistence across cycles" begin
  color = attachment_resource(device, nothing; format = RGBA{Float16}, usage_flags = Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_TRANSFER_DST_BIT | Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT, dims = [1920, 1080])
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
  draw = draw_texture(device, vdata, color)
  hashes = UInt64[]
  for i in 1:5
    data = render_graphics(device, draw)
    push!(hashes, hash(data))
  end
  @test all(==(0x7c14e3ffe5603da5), hashes)
end;
