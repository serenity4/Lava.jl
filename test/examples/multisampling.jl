function program_3(device, vdata, color)
  vert_interface = ShaderInterface(
    storage_classes = [SPIRV.StorageClassOutput, SPIRV.StorageClassOutput, SPIRV.StorageClassInput, SPIRV.StorageClassPushConstant],
    variable_decorations = dictionary([
      1 => dictionary([SPIRV.DecorationLocation => [0U]]),
      2 => dictionary([SPIRV.DecorationBuiltIn => [SPIRV.BuiltInPosition]]),
      3 => dictionary([SPIRV.DecorationBuiltIn => [SPIRV.BuiltInVertexIndex]]),
    ]),
    features = device.spirv_features,
  )

  frag_interface = ShaderInterface(
    execution_model = SPIRV.ExecutionModelFragment,
    storage_classes = [SPIRV.StorageClassOutput, SPIRV.StorageClassInput],
    variable_decorations = dictionary([
      1 => dictionary([SPIRV.DecorationLocation => [0U]]),
      2 => dictionary([SPIRV.DecorationLocation => [0U]]),
    ]),
    features = device.spirv_features,
  )

  vert_shader = @shader vert_interface rectangle_vert(::Vec{4,Float32}, ::Vec{4,Float32}, ::UInt32, ::DrawData)
  frag_shader = @shader frag_interface rectangle_frag(::Vec{4,Float32}, ::Vec{4,Float32})
  prog = Program(device, vert_shader, frag_shader)

  rg = RenderGraph(device)

  graphics = RenderNode(render_area = RenderArea(1920, 1080), stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT | Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT) do rec
    set_program(rec, prog)
    draw(rec, vdata, collect(1:3), color; alignment = 4)
  end

  @add_resource_dependencies rg begin
    (color * 4 => (0.08, 0.05, 0.1, 1.0))::Color = graphics()
  end
end

@testset "Multisampled triangle" begin
  color_ms = attachment(device, Vk.FORMAT_R16G16B16A16_SFLOAT; samples = 4, usage = Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_TRANSFER_DST_BIT | Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT, dims = (1920, 1080))
  pcolor_ms = PhysicalAttachment(color_ms)
  vdata = [
    (0.0f0, 0.8f0, RGB{Float32}(1.0, 0.0, 0.0)),
    (0.5f0, -0.8f0, RGB{Float32}(0.0, 0.0, 1.0)),
    (-0.5f0, -0.8f0, RGB{Float32}(0.0, 1.0, 0.0)),
  ]
  rg = program_3(device, vdata, pcolor_ms)

  @test wait(render(rg))
  data = collect(RGBA{Float16}, color_ms.view.image, device)
  save_test_render("triangle_multisampled.png", data, 0x4b29f98dcdacc431)
end
