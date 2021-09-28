is_ci = get(ENV, "JULIA_CI", "false") == "true"
if is_ci
    import SwiftShader_jll
    ENV["JULIA_VULKAN_LIBNAME"] = basename(SwiftShader_jll.libvulkan)
end

using Lava

is_ci && Vk.@set_driver :SwiftShader
using Test

resource(filename) = joinpath(@__DIR__, "resources", filename)

@testset "Lava.jl" begin
    instance, device = init(; with_validation = !is_ci)

    @testset "Buffers & Memory" begin
        buffer = BufferBlock(device, 100)
        @test !isallocated(buffer)

        sbuffer = similar(buffer, MEMORY_DOMAIN_DEVICE)
        @test sbuffer.size == buffer.size
        @test sbuffer.sharing_mode == buffer.sharing_mode
        @test sbuffer.queue_family_indices == buffer.queue_family_indices
        @test sbuffer.usage == buffer.usage
        @test !isallocated(sbuffer)

        subbuffer = @view buffer[2:4:end]
        @test subbuffer isa SubBuffer
        @test subbuffer.offset == 2
        @test subbuffer.stride == 4
        @test subbuffer.size == 96
        @test_throws UndefRefError memory(subbuffer)

        subbuffer = @view buffer[2:end]
        @test subbuffer isa SubBuffer
        @test subbuffer.offset == 2
        @test subbuffer.stride == 0
        @test subbuffer.size == size(buffer) - 2

        mem = MemoryBlock(device, 100, 7, MEMORY_DOMAIN_HOST_CACHED)
        submem = @view mem[2:4]
        @test submem isa SubMemory
        redirect_stderr(devnull) do
            mem_toomuch = Lava.memory_block(device, 100000000000000000, 7, MEMORY_DOMAIN_DEVICE)
            @test iserror(mem_toomuch)
            @test unwrap_error(mem_toomuch).code == Vk.ERROR_OUT_OF_DEVICE_MEMORY
        end

        unwrap(allocate!(buffer, MEMORY_DOMAIN_HOST_CACHED))
        @test isallocated(buffer)
        @test device_address(buffer) ≠ C_NULL
        @test device_address(subbuffer) == device_address(buffer) + subbuffer.offset
        @test memory(buffer) isa MemoryBlock
        @test memory(subbuffer) isa SubMemory
        mem2 = MemoryBlock(device, 1000, 7, MEMORY_DOMAIN_HOST_CACHED)
        buffer2 = BufferBlock(device, 100)
        unwrap(bind!(buffer2, mem2))
        @test memory(buffer2) === mem2

        sbuffer = similar(buffer, MEMORY_DOMAIN_DEVICE)
        @test isallocated(sbuffer)
        @test memory(sbuffer) ≠ memory(buffer)

        @test Lava.buffer(device, collect(1:1000)) isa Buffer

        @testset "Allocators" begin
            la = Lava.LinearAllocator(device, 1000)
            @test device_address(la) ≠ C_NULL

            sub = copyto!(la, [1, 2, 3])
            @test Lava.offset(sub) == 0
            @test size(sub) == 24
            sub = copyto!(la, (4f0, 5f0, 6f0))
            @test Lava.offset(sub) == 24
            @test size(sub) == 12
            sub = copyto!(la, (4f0, 5f0, 6f0))
            # 8-byte alignment requirement
            @test Lava.offset(sub) == 40

            Lava.reset!(la)
        end
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
        include("frame_graph.jl")
    end

    @testset "Shaders" begin
        include("shaders.jl")
    end
end
