function get_dependency_infos!(rg, nodes)
  init = (Vk.DependencyInfo[], SynchronizationState())
  foldl(((infos, state), node) -> (push!(infos, dependency_info!(state, rg, node)), state), nodes; init)
end

@testset "Node synchronization" begin
  color = attachment_resource(device, zeros(RGBA{Float32}, 16, 16); usage_flags = Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
  normal = image_resource(device, zeros(RGBA{Float32}, 16, 16))
  depth = attachment_resource(device, zeros(Float32, 16, 16); format = Vk.FORMAT_D32_SFLOAT, usage_flags = Vk.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, aspect = Vk.IMAGE_ASPECT_DEPTH_BIT)
  final = attachment_resource(device, zeros(RGBA{Float32}, 16, 16); usage_flags = Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
  nodes = [
    RenderNode(render_area = RenderArea(1920, 1080), stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT | Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT),
    RenderNode(render_area = RenderArea(1920, 1080), stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT | Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT),
  ]
  push!(nodes[1].commands, fake_graphics_command(RenderTargets(color; depth)))
  push!(nodes[2].commands, fake_graphics_command(RenderTargets(color, final; depth)))
  rg = RenderGraph(device)
  @add_resource_dependencies rg begin
    (color => (0.0, 0.0, 0.0, 1.0))::Color, depth::Depth = nodes[1](normal::Texture)
    final::Color = nodes[2](color::Color, depth::Depth)
  end
  bake!(rg)
  (dependency_infos, state) = get_dependency_infos!(rg, nodes)
  info = dependency_infos[1]
  @test length(info.buffer_memory_barriers) == 0
  @test length(info.image_memory_barriers) == 3
  barriers = info.image_memory_barriers
  @test barriers[1].image == normal.data.handle
  @test barriers[2].image == color.data.view.image.handle
  @test barriers[3].image == depth.data.view.image.handle
  @test all(iszero, [barrier.src_access_mask for barrier in barriers])
  @test all(iszero, [barrier.src_stage_mask for barrier in barriers])
  @test barriers[1].dst_stage_mask == Int(nodes[1].stages)
  @test barriers[2].dst_stage_mask == Int(Vk.PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)
  @test barriers[3].dst_stage_mask == Int(Vk.PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | Vk.PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)
  @test barriers[1].dst_access_mask == Int(Vk.ACCESS_2_SHADER_READ_BIT)
  @test barriers[2].dst_access_mask == Int(Vk.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT)
  @test barriers[3].dst_access_mask == Int(Vk.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)

  info = dependency_infos[2]
  @test length(info.buffer_memory_barriers) == 0
  @test length(info.image_memory_barriers) == 3
  barriers = info.image_memory_barriers
  @test barriers[1].image == color.data.view.image.handle
  @test barriers[2].image == depth.data.view.image.handle
  @test barriers[3].image == final.data.view.image.handle
  @test barriers[1].src_stage_mask == Int(Vk.PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)
  @test barriers[2].src_stage_mask == Int(Vk.PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | Vk.PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)
  @test barriers[3].src_stage_mask == 0
  @test barriers[1].dst_stage_mask == Int(Vk.PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)
  @test barriers[2].dst_stage_mask == Int(Vk.PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | Vk.PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)
  @test barriers[3].dst_stage_mask == Int(Vk.PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)
  @test barriers[1].src_access_mask == Int(Vk.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT)
  @test barriers[2].src_access_mask == Int(Vk.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
  @test barriers[1].dst_access_mask == Int(Vk.ACCESS_2_COLOR_ATTACHMENT_READ_BIT)
  @test barriers[2].dst_access_mask == Int(Vk.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT)
  @test iszero(barriers[3].src_access_mask)
  @test iszero(barriers[3].src_stage_mask)

  finish!(rg)
end;
