@testset "Graphics pipelines" begin
  vertex_shader = @vertex device.spirv_features test_shader(::Output{Position}::Vec{4,Float32})
  fragment_shader = @fragment device.spirv_features test_shader(::Output::Vec{4,Float32})
  program = Program(device, vertex_shader, fragment_shader)
  color = PhysicalAttachment(attachment(device, format = Lava.format(BGRA{N0f8}), dims = (1920, 1080)))
  rdes = PhysicalDescriptors(device)

  # Clear previous state.
  empty!(device.pending_pipelines)
  empty!(device.pipeline_ht)
  empty!(device.pipeline_layouts)
  empty!(device.pipeline_layout_ht)

  info = Lava.pipeline_info(device, RenderArea(1920, 1080).rect, program, RenderState(), ProgramInvocationState(), rdes, RenderTargets([color]))
  hash = Lava.request_pipeline(device, info)
  Lava.create_pipelines(device)
  pipeline = device.pipeline_ht[hash]
  @test pipeline isa Lava.Pipeline
  @test length(device.pipeline_ht) == 1

  info2 = Lava.pipeline_info(device, RenderArea(1920, 1080).rect, program, RenderState(), ProgramInvocationState(), rdes, RenderTargets([color]))
  hash2 = Lava.request_pipeline(device, info)
  @test hash2 == hash
  Lava.create_pipelines(device)
  @test length(device.pipeline_ht) == 1
  @test device.pipeline_ht[hash] === pipeline
end
