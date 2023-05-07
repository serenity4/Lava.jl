@testset "Pipelines" begin
  layout = Lava.pipeline_layout(device)

  @testset "Graphics pipelines" begin
    vertex_shader = @vertex device test_shader(::Vec{4,Float32}::Output{Position})
    fragment_shader = @fragment device test_shader(::Vec{4,Float32}::Output)
    program = Program(vertex_shader, fragment_shader)
    color = attachment_resource(device, nothing; format = BGRA{N0f8}, dims = [1920, 1080])

    # Clear previous state.
    empty!(device.pending_pipelines_graphics)
    empty!(device.pipeline_ht_graphics)

    info = Lava.pipeline_info_graphics(RenderArea(1920, 1080), program, RenderState(), ProgramInvocationState(), RenderTargets([color]), layout, Dict())
    h = Lava.request_pipeline(device, info)
    Lava.create_pipelines!(device)
    pipeline = device.pipeline_ht_graphics[h]
    @test pipeline isa Lava.Pipeline
    @test length(device.pipeline_ht_graphics) == 1

    info2 = Lava.pipeline_info_graphics(RenderArea(1920, 1080), program, RenderState(), ProgramInvocationState(), RenderTargets([color]), layout, Dict())
    h2 = Lava.request_pipeline(device, info)
    @test h2 == h
    Lava.create_pipelines!(device)
    @test length(device.pipeline_ht_graphics) == 1
    @test device.pipeline_ht_graphics[h] === pipeline
  end

  @testset "Compute pipelines" begin
    compute_shader = @compute device (() -> nothing)()
    program = Program(compute_shader)

    # Clear previous state.
    empty!(device.pending_pipelines_compute)
    empty!(device.pipeline_ht_compute)

    info = Lava.pipeline_info_compute(program, layout)
    h = Lava.request_pipeline(device, info)
    Lava.create_pipelines!(device)
    pipeline = device.pipeline_ht_compute[h]
    @test pipeline isa Lava.Pipeline
    @test length(device.pipeline_ht_compute) == 1

    info2 = Lava.pipeline_info_compute(program, layout)
    h2 = Lava.request_pipeline(device, info)
    @test h2 == h
    Lava.create_pipelines!(device)
    @test length(device.pipeline_ht_compute) == 1
    @test device.pipeline_ht_compute[h] === pipeline
  end
end;
