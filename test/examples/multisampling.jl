function multisampling_invocation(device, vdata, color; prog = rectangle_program(device))
  invocation_data = @invocation_data @block vdata
  ProgramInvocation(
    prog,
    DrawIndexed(1:3),
    RenderTargets(color),
    invocation_data,
    RenderState(),
    ProgramInvocationState(),
    @resource_dependencies begin
      @write
      (color * 4 => (0.08, 0.05, 0.1, 1.0))::Color
    end
  )
end

@testset "Multisampled triangle" begin
  color_ms = attachment_resource(device, nothing; format = Vk.FORMAT_R16G16B16A16_SFLOAT, samples = 4, usage_flags = Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_TRANSFER_DST_BIT | Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT, dims = [1920, 1080])
  vdata = [
    PosColor(Vec2(0.0, 0.8), Arr{Float32}(1.0, 0.0, 0.0)),
    PosColor(Vec2(0.5, -0.8), Arr{Float32}(0.0, 0.0, 1.0)),
    PosColor(Vec2(-0.5, -0.8), Arr{Float32}(0.0, 1.0, 0.0)),
  ]
  invocation = multisampling_invocation(device, vdata, color_ms)
  data = render_graphics(device, graphics_node(invocation))
  save_test_render("triangle_multisampled.png", data, 0x4b29f98dcdacc431)
end;
