using Graphs: nv, ne

@testset "Building a render graph" begin
  rg = RenderGraph(device)

  # Specify resources used for the rendering process.
  # Resources can either be physical (e.g. be allocated from a `device` and be bound to GPU memory)
  # or logical (the resource will be created lazily, enabling optimizations such as aliasing or
  # non-allocation of unused resources).

  # Logical resource.
  vbuffer = buffer(rg, 1024)
  @test vbuffer isa LogicalBuffer
  # Regular `BufferBlock`.
  ibuffer = wait(buffer(device, collect(1:100); usage = Vk.BUFFER_USAGE_INDEX_BUFFER_BIT))
  @test ibuffer isa BufferBlock

  # Logical resources all the way.
  average_luminance = image(rg, Vk.FORMAT_R32G32B32A32_SFLOAT, (16, 16))
  emissive = attachment(rg, Vk.FORMAT_R32G32B32A32_SFLOAT)
  albedo = attachment(rg, Vk.FORMAT_R32G32B32A32_SFLOAT)
  normal = attachment(rg, Vk.FORMAT_R32G32B32A32_SFLOAT)
  pbr = attachment(rg, Vk.FORMAT_R32G32B32A32_SFLOAT)
  color = attachment(rg, Vk.FORMAT_R32G32B32A32_SFLOAT; samples = 4)
  output = attachment(rg, Vk.FORMAT_R32G32B32A32_SFLOAT)
  depth = attachment(rg, Vk.FORMAT_D32_SFLOAT)
  shadow_main = image(rg, Vk.FORMAT_D32_SFLOAT, (16, 16))
  shadow_near = image(rg, Vk.FORMAT_D32_SFLOAT, (16, 16))
  bloom_downsample_3 = image(rg, Vk.FORMAT_R32G32B32A32_SFLOAT, (16, 16))

  # No rendering done here, doesn't matter what function is passed. We pass in `identity`.
  transfer = RenderNode(identity) # unused
  gbuffer = RenderNode(
    identity;
    render_area = RenderArea(1920, 1080),
    stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT | Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
  )
  lighting = RenderNode(identity; render_area = RenderArea(1920, 1080), stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT)
  adapt_luminance = RenderNode(identity, render_area = RenderArea(1920, 1080), stages = Vk.PIPELINE_STAGE_2_ALL_GRAPHICS_BIT)
  combine = RenderNode(identity, render_area = RenderArea(1920, 1080), stages = Vk.PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)

  @add_resource_dependencies rg begin
    emissive::Color, albedo::Color, normal::Color, pbr::Color, depth::Depth = gbuffer(vbuffer::Buffer::Vertex, ibuffer::Buffer::Index)
    color::Color = lighting(emissive::Color, albedo::Color, normal::Color, pbr::Color, depth::Depth, shadow_main::Texture, shadow_near::Texture)
    average_luminance::Image::Storage = adapt_luminance(average_luminance::Image::Storage, bloom_downsample_3::Texture)
    output::Color = combine(color::Color, average_luminance::Texture)
  end

  @test nv(rg.resource_graph) == 4 + 13
  @test ne(rg.resource_graph) == 6 + 8 + 3 + 3

  uses = Lava.ResourceUses(rg)

  usage = uses[color]
  @test usage.type == RESOURCE_TYPE_COLOR_ATTACHMENT
  @test usage.access == WRITE | READ
  @test usage.stages == Vk.PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT
  @test usage.aspect == Vk.IMAGE_ASPECT_COLOR_BIT

  usage = uses[depth]
  @test usage.type == RESOURCE_TYPE_DEPTH_ATTACHMENT
  @test usage.aspect == Vk.IMAGE_ASPECT_DEPTH_BIT

  @test isnothing(Lava.check_physical_resources(rg, uses))

  uses2 = deepcopy(uses)
  ibuffer_uuid = findfirst(x -> Vk.BUFFER_USAGE_INDEX_BUFFER_BIT in x.usage, rg.physical_resources.buffers)
  ibuffer_usage = uses2[ibuffer_uuid]
  set!(uses2.buffers, ibuffer_uuid, @set ibuffer_usage.usage = Vk.BUFFER_USAGE_STORAGE_BUFFER_BIT)
  @test_throws ErrorException Lava.check_physical_resources(rg, uses2)

  resources = Lava.materialize_logical_resources(rg, uses)
  @test resources isa Lava.PhysicalResources
end
