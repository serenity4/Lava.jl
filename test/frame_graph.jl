@testset "Building a frame graph" begin
  frame = Frame(device)
  fg = FrameGraph(device, frame)

  add_pass!(identity, fg, :gbuffer, RenderPass((0, 0, 1920, 1080)))
  add_pass!(identity, fg, :lighting, RenderPass((0, 0, 1920, 1080)))
  add_pass!(identity, fg, :adapt_luminance, RenderPass((0, 0, 1920, 1080)))
  add_pass!(identity, fg, :combine, RenderPass((0, 0, 1920, 1080)))
  # can't add a pass more than once
  @test_throws ErrorException add_pass!(identity, fg, :combine, RenderPass((0, 0, 1920, 1080)))

  add_resource!(fg, :vbuffer, BufferResourceInfo(1024))
  add_resource!(fg, :ibuffer, BufferResourceInfo(1024))
  add_resource!(fg, :average_luminance, ImageResourceInfo(Vk.FORMAT_R32G32B32A32_SFLOAT))
  add_resource!(fg, :emissive, AttachmentResourceInfo(Vk.FORMAT_R32G32B32A32_SFLOAT))
  add_resource!(fg, :albedo, AttachmentResourceInfo(Vk.FORMAT_R32G32B32A32_SFLOAT))
  add_resource!(fg, :normal, AttachmentResourceInfo(Vk.FORMAT_R32G32B32A32_SFLOAT))
  add_resource!(fg, :pbr, AttachmentResourceInfo(Vk.FORMAT_R32G32B32A32_SFLOAT))
  add_resource!(fg, :color, AttachmentResourceInfo(Vk.FORMAT_R32G32B32A32_SFLOAT))
  add_resource!(fg, :output, AttachmentResourceInfo(Vk.FORMAT_R32G32B32A32_SFLOAT))
  add_resource!(fg, :depth, AttachmentResourceInfo(Vk.FORMAT_D32_SFLOAT))
  # can't add a resource more than once
  @test_throws ErrorException add_resource!(fg, :depth, AttachmentResourceInfo(Vk.FORMAT_D32_SFLOAT))

  # imported
  add_resource!(fg, :shadow_main, AttachmentResourceInfo(Vk.FORMAT_D32_SFLOAT))
  add_resource!(fg, :shadow_near, AttachmentResourceInfo(Vk.FORMAT_D32_SFLOAT))
  add_resource!(fg, :bloom_downsample_3, AttachmentResourceInfo(Vk.FORMAT_R32G32B32A32_SFLOAT))

  usages = @resource_usages begin
    emissive::Color, albedo::Color, normal::Color, pbr::Color, depth::Depth = gbuffer(vbuffer::Buffer::Vertex, ibuffer::Buffer::Index)
    color::Color = lighting(emissive::Color, albedo::Color, normal::Color, pbr::Color, depth::Depth, shadow_main::Texture, shadow_near::Texture)
    average_luminance::Image::Storage = adapt_luminance(average_luminance::Image::Storage, bloom_downsample_3::Texture)
    output::Color = combine(color::Color, average_luminance::Texture)
  end

  add_resource_usage!(fg, usages)
  Lava.resolve_attributes!(fg)

  @test Lava.buffer_usage(fg, :vbuffer) == Vk.BUFFER_USAGE_VERTEX_BUFFER_BIT
  @test Lava.image_usage(fg, :depth) == Vk.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
  @test Lava.image_usage(fg, :shadow_main) == Vk.IMAGE_USAGE_SAMPLED_BIT

  for resource in keys(fg.resources)
    @test Int(Lava.resource_attribute(fg, resource, :usage)) ≠ 0
  end
end
