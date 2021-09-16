is_ci = get(ENV, "JULIA_CI", "false") == "true"
if is_ci
    import SwiftShader_jll
    ENV["JULIA_VULKAN_LIBNAME"] = basename(SwiftShader_jll.libvulkan)
end

using Lava

is_ci && Vk.@set_driver :SwiftShader
using Test

@testset "Lava.jl" begin
    instance, device = init(; with_validation = !is_ci)

    @testset "Buffers & Memory" begin
        buffer = BufferBlock(device, 100, Vk.BUFFER_USAGE_TRANSFER_DST_BIT)
        @test !isallocated(buffer)
        subbuffer = @view buffer[2:4:end]
        @test subbuffer isa SubBuffer
        @test_throws UndefRefError memory(subbuffer)

        mem = unwrap(MemoryBlock(device, 100, 7, MEMORY_DOMAIN_HOST_CACHED))
        submem = @view mem[2:4]
        @test submem isa SubMemory
        redirect_stderr(devnull) do
            mem_toomuch = MemoryBlock(device, 100000000000000000, 7, MEMORY_DOMAIN_DEVICE)
            @test iserror(mem_toomuch)
            @test unwrap_error(mem_toomuch).code == Vk.ERROR_OUT_OF_DEVICE_MEMORY
        end

        unwrap(allocate!(buffer, MEMORY_DOMAIN_HOST_CACHED))
        @test isallocated(buffer)
        @test memory(buffer) isa MemoryBlock
        @test memory(subbuffer) isa SubMemory
        mem2 = unwrap(MemoryBlock(device, 1000, 7, MEMORY_DOMAIN_HOST_CACHED))
        buffer2 = BufferBlock(device, 100, Vk.BUFFER_USAGE_TRANSFER_DST_BIT)
        unwrap(bind!(buffer2, mem2))
        @test memory(buffer2) === mem2
    end

    @testset "Images" begin
        image = ImageBlock(device, (512, 512), Vk.FORMAT_R32G32B32A32_SFLOAT, Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        @test !isallocated(image)
        allocate!(image, MEMORY_DOMAIN_DEVICE)
        @test isallocated(image)
        @test memory(image) isa MemoryBlock
        v = View(image)
        @test v isa ImageView
    end

    @testset "Frame Graph" begin
        fg = FrameGraph(device)

        add_pass!(fg, :gbuffer; clear_values = (0.1, 0.01, 0.08, 1.))
        add_pass!(fg, :lighting; clear_values = (0.1, 0.01, 0.08, 1.))
        add_pass!(fg, :adapt_luminance; clear_values = (0.1, 0.01, 0.08, 1.))
        add_pass!(fg, :combine; clear_values = (0.1, 0.01, 0.08, 1.))
        # can't add a pass more than once
        @test_throws ErrorException add_pass!(fg, :combine; clear_values = (0.1, 0.01, 0.08, 1.))

        add_resource!(fg, :vbuffer, BufferResourceInfo(1024))
        add_resource!(fg, :ibuffer, BufferResourceInfo(1024))
        add_resource!(fg, :average_luminance, BufferResourceInfo(2048))
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
            average_luminance::Buffer::Storage = adapt_luminance(average_luminance::Buffer::Storage, bloom_downsample_3::Texture)
            output::Color = combine(color::Color, average_luminance::Texture)
        end

        #=

        # or, without a macro:
        passes = [
            Pass(
                # pass name
                :gbuffer,
                # reads
                [:vbuffer => RESOURCE_TYPE_VERTEX_BUFFER, :ibuffer => RESOURCE_TYPE_INDEX_BUFFER],
                # writes
                [
                    :emissive => RESOURCE_TYPE_COLOR_ATTACHMENT, :albedo => RESOURCE_TYPE_COLOR_ATTACHMENT,
                    :normal => RESOURCE_TYPE_COLOR_ATTACHMENT, :pbr => RESOURCE_TYPE_COLOR_ATTACHMENT,
                    :depth => RESOURCE_TYPE_DEPTH_ATTACHMENT,
                ]
            ),
            ... # other passes
        ]

        =#

        add_resource_usage!(fg, usages)
    end

    @testset "Shaders" begin
        # include("shaders.jl")
    end
end
