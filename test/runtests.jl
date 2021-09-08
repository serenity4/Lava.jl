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
        mem_toomuch = MemoryBlock(device, 100000000000000000, 7, MEMORY_DOMAIN_DEVICE)
        @test iserror(mem_toomuch)
        @test unwrap_error(mem_toomuch).code == Vk.ERROR_OUT_OF_DEVICE_MEMORY

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
end
