@testset "Building a frame graph" begin
  rg = RenderGraph(device)

  # Specify resources used for the rendering process.
  # Resources can either be physical (e.g. be allocated from a `device` and be bound to GPU memory)
  # or logical (the resource will be created lazily, enabling optimizations such as aliasing or
  # non-allocation of unused resources).
  resources = (;
    # Logical resource.
    vbuffer = buffer(rg, 1024),
    # Physical resource, allocated and returned as a `BufferBlock`.
    ibuffer = buffer(device, collect(1:100)),
    average_luminance = image(rg, Vk.FORMAT_R32G32B32A32_SFLOAT),
    emissive = attachment(rg, Vk.FORMAT_R32G32B32A32_SFLOAT),
    albedo = attachment(rg, Vk.FORMAT_R32G32B32A32_SFLOAT),
    normal = attachment(rg, Vk.FORMAT_R32G32B32A32_SFLOAT),
    pbr = attachment(rg, Vk.FORMAT_R32G32B32A32_SFLOAT),
    color = attachment(rg, Vk.FORMAT_R32G32B32A32_SFLOAT),
    output = attachment(rg, Vk.FORMAT_R32G32B32A32_SFLOAT),
    depth = attachment(rg, Vk.FORMAT_D32_SFLOAT),
    shadow_main = attachment(rg, Vk.FORMAT_D32_SFLOAT),
    shadow_near = attachment(rg, Vk.FORMAT_D32_SFLOAT),
    bloom_downsample_3 = attachment(rg, Vk.FORMAT_R32G32B32A32_SFLOAT),
  )

  nodes = (;
    transfer = render_node(rg),
    gbuffer = render_pass(rg, (0, 0, 1920, 1080)),
    lighting = render_pass(rg, (0, 0, 1920, 1080)),
    adapt_luminance = render_pass(rg, (0, 0, 1920, 1080)),
    combine = render_pass(rg, (0, 0, 1920, 1080)),
  )

  # `nodes` and `resources` optionally provided as namespaces. Providing none of them will just use the local scope.
  usages = @resource_dependencies nodes resources begin
    emissive::Color, albedo::Color, normal::Color, pbr::Color, depth::Depth = gbuffer(vbuffer::Buffer::Vertex, ibuffer::Buffer::Index)
    color::Color = lighting(emissive::Color, albedo::Color, normal::Color, pbr::Color, depth::Depth, shadow_main::Texture, shadow_near::Texture)
    average_luminance::Image::Storage = adapt_luminance(average_luminance::Image::Storage, bloom_downsample_3::Texture)
    output::Color = combine(color::Color, average_luminance::Texture)
  end

  add_resource_dependencies!(rg, usages)
  Lava.resolve_attributes!(rg)

  @test Lava.buffer_usage(rg, resources.vbuffer) == Vk.BUFFER_USAGE_VERTEX_BUFFER_BIT
  @test Lava.image_usage(rg, resources.depth) == Vk.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
  @test Lava.image_usage(rg, resources.shadow_main) == Vk.IMAGE_USAGE_SAMPLED_BIT

  for uuid in Lava.resource_uuids(rg)
    @test !iszero(Int(Lava.resource_attribute(rg, uuid, :usage)))
  end
end
