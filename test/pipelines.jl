@testset "Graphics pipelines" begin
  vertex_shader = @vertex device test_shader(::Vec{4,Float32}::Output{Position})
  fragment_shader = @fragment device test_shader(::Vec{4,Float32}::Output)
  program = Program(device, vertex_shader, fragment_shader)
  color = attachment_resource(device, nothing; format = Lava.format(BGRA{N0f8}), dims = [1920, 1080])

  # Clear previous state.
  empty!(device.pending_pipelines)
  empty!(device.pipeline_ht)
  empty!(device.pipeline_layouts)
  empty!(device.pipeline_layout_ht)

  info = Lava.pipeline_info(device, RenderArea(1920, 1080), program, RenderState(), ProgramInvocationState(), RenderTargets([color]))
  hash = Lava.request_pipeline(device, info)
  Lava.create_pipelines(device)
  pipeline = device.pipeline_ht[hash]
  @test pipeline isa Lava.Pipeline
  @test length(device.pipeline_ht) == 1

  info2 = Lava.pipeline_info(device, RenderArea(1920, 1080), program, RenderState(), ProgramInvocationState(), RenderTargets([color]))
  hash2 = Lava.request_pipeline(device, info)
  @test hash2 == hash
  Lava.create_pipelines(device)
  @test length(device.pipeline_ht) == 1
  @test device.pipeline_ht[hash] === pipeline
end;
