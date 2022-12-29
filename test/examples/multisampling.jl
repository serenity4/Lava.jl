function program_3(device, vdata, color)
  rg = RenderGraph(device)

  graphics = RenderNode(render_area = RenderArea(1920, 1080), stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT | Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)

  @add_resource_dependencies rg begin
    (color * 4 => (0.08, 0.05, 0.1, 1.0))::Color = graphics()
  end

  rec = StatefulRecording()
  set_program(rec, rectangle_program(device))
  set_data(rec, rg, vdata)
  draw(graphics, rec, collect(1:3), color)
  rg
end

@testset "Multisampled triangle" begin
  color_ms = attachment_resource(device, nothing; format = Vk.FORMAT_R16G16B16A16_SFLOAT, samples = 4, usage_flags = Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_TRANSFER_DST_BIT | Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT, dims = [1920, 1080])
  vdata = [
    PosColor(Vec2(0.0, 0.8), Arr{Float32}(1.0, 0.0, 0.0)),
    PosColor(Vec2(0.5, -0.8), Arr{Float32}(0.0, 0.0, 1.0)),
    PosColor(Vec2(-0.5, -0.8), Arr{Float32}(0.0, 1.0, 0.0)),
  ]
  rg = program_3(device, vdata, color_ms)

  render!(rg)
  data = read_data(device, color_ms)
  save_test_render("triangle_multisampled.png", data, 0x4b29f98dcdacc431)
end;
