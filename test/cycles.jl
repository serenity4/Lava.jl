function invocation_cycles(device, vdata, pcolor, prog = rectangle_program(device))
  invocation_data = @invocation_data begin
    @block vdata
  end
  ProgramInvocation(
    prog,
    DrawIndexed(1:4),
    RenderTargets(pcolor),
    invocation_data,
    RenderState(),
    setproperties(ProgramInvocationState(), (;
      primitive_topology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
      triangle_orientation = Vk.FRONT_FACE_COUNTER_CLOCKWISE,
    )),
  )
end

@testset "Data persistence across cycles" begin
  color = attachment(device; format = Vk.FORMAT_R16G16B16A16_SFLOAT, usage = Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_TRANSFER_DST_BIT | Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT, dims = (1920, 1080))
  pcolor = PhysicalAttachment(color)
  vdata = [
    PosColor(Vec2(-0.5, 0.5), Arr{Float32}(1.0, 0.0, 0.0)),
    PosColor(Vec2(-0.5, -0.5), Arr{Float32}(0.0, 1.0, 0.0)),
    PosColor(Vec2(0.5, 0.5), Arr{Float32}(1.0, 1.0, 1.0)),
    PosColor(Vec2(0.5, -0.5), Arr{Float32}(0.0, 0.0, 1.0)),
  ]
  invocation = invocation_cycles(device, vdata, pcolor)
  graphics = RenderNode(render_area = RenderArea(1920, 1080), stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT | Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)
  push!(graphics.program_invocations, invocation)

  hashes = UInt64[]
  for i in 1:5
    rg = RenderGraph(device)
    # Collect the previous render graph.
    GC.gc()
    @add_resource_dependencies rg begin
      (pcolor => (0.08, 0.05, 0.1, 1.0))::Color = graphics()
    end
    render(rg)
    push!(hashes, hash(collect(RGBA{Float16}, color.view.image, device)))
  end
  @test all(==(0x9430efd8e0911300), hashes)
end;
