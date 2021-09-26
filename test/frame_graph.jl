instance, device = init(; with_validation = !is_ci)

@testset "Building a frame graph" begin
    fg = FrameGraph(device)

    add_pass!(identity, fg, :gbuffer; clear_values = (0.1, 0.01, 0.08, 1.))
    add_pass!(identity, fg, :lighting; clear_values = (0.1, 0.01, 0.08, 1.))
    add_pass!(identity, fg, :adapt_luminance; clear_values = (0.1, 0.01, 0.08, 1.))
    add_pass!(identity, fg, :combine; clear_values = (0.1, 0.01, 0.08, 1.))
    # can't add a pass more than once
    @test_throws ErrorException add_pass!(identity, fg, :combine; clear_values = (0.1, 0.01, 0.08, 1.))

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

@testset "Rendering" begin
    fg = FrameGraph(device)
    prog = Program(device, ShaderSpecification(resource("headless.vert"), GLSL), ShaderSpecification(resource("headless.frag"), GLSL))
    add_pass!(fg, :main; clear_values = (0.1, 0.1, 0.1, 1.)) do (rec, fg)
        set_program(rec, prog)
    end
end
