is_ci = get(ENV, "JULIA_CI", "false") == "true"
if is_ci
    import SwiftShader_jll
    ENV["JULIA_VULKAN_LIBNAME"] = basename(SwiftShader_jll.libvulkan)
end

using Lava

is_ci && Vk.@set_driver :SwiftShader
using Test
using GeometryExperiments
using Colors
using FileIO
using ImageIO
using Accessors

resource(filename) = joinpath(@__DIR__, "resources", filename)

instance, device = init(; with_validation = !is_ci, device_specific_features = [:shader_int_64, :sampler_anisotropy])

@testset "Lava.jl" begin
    @testset "Buffers & Memory" begin
        b = BufferBlock(device, 100)
        @test !isallocated(b)

        sb = similar(b)
        @test sb.size == b.size
        @test sb.sharing_mode == b.sharing_mode
        @test sb.queue_family_indices == b.queue_family_indices
        @test sb.usage == b.usage
        @test !isallocated(sb)

        sub = @view b[2:4:end]
        @test sub isa SubBuffer
        @test sub.offset == 2
        @test sub.stride == 4
        @test sub.size == 96
        @test_throws UndefRefError memory(sub)

        sub = @view b[2:end]
        @test sub isa SubBuffer
        @test sub.offset == 2
        @test sub.stride == 0
        @test sub.size == size(b) - 2

        mem = MemoryBlock(device, 100, 7, MEMORY_DOMAIN_HOST_CACHED)
        submem = @view mem[2:4]
        @test submem isa SubMemory
        redirect_stderr(devnull) do
            too_much = Lava.memory_block(device, 100000000000000000, 7, MEMORY_DOMAIN_DEVICE)
            @test iserror(too_much)
            @test unwrap_error(too_much).code == Vk.ERROR_OUT_OF_DEVICE_MEMORY
            yield()
        end

        allocate!(b, MEMORY_DOMAIN_HOST_CACHED)
        @test isallocated(b)
        @test device_address(b) ≠ C_NULL
        @test device_address(sub) == device_address(b) + sub.offset
        @test memory(b) isa MemoryBlock
        @test memory(sub) isa SubMemory
        mem2 = MemoryBlock(device, 1000, 7, MEMORY_DOMAIN_HOST_CACHED)
        b2 = BufferBlock(device, 100)
        bind!(b2, mem2)
        @test memory(b2) === mem2

        sb = similar(b, memory_domain = MEMORY_DOMAIN_DEVICE)
        @test isallocated(sb)
        @test memory(sb) ≠ memory(b)

        @test buffer(device, collect(1:1000)) isa Buffer

        @testset "Allocators" begin
            la = Lava.LinearAllocator(device, 1000)
            @test device_address(la) ≠ C_NULL

            sub = copyto!(la, [1, 2, 3])
            @test offset(sub) == 0
            @test size(sub) == 24
            sub = copyto!(la, (4f0, 5f0, 6f0))
            @test offset(sub) == 24
            @test size(sub) == 12
            sub = copyto!(la, (4f0, 5f0, 6f0))
            # 8-byte alignment requirement
            @test offset(sub) == 40

            Lava.reset!(la)
        end

        @testset "Data transfer" begin
            b1 = buffer(device, collect(1:1000); usage = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT)
            b2 = BufferBlock(device, 8000; usage = Vk.BUFFER_USAGE_TRANSFER_DST_BIT | Vk.BUFFER_USAGE_TRANSFER_SRC_BIT)
            allocate!(b2, MEMORY_DOMAIN_DEVICE)
            @test reinterpret(Int, collect(b1)) == collect(1:1000)
            t = transfer(device, b1, b2; signal_fence = true)
            @test t isa ExecutionState
            @test wait(t)
            @test reinterpret(Int, collect(b2, device)) == collect(1:1000)

            b3, exec = buffer(device, collect(1:1000), Val(true); usage = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT)
            wait(exec)
            @test reinterpret(Int, collect(b3, device)) == collect(1:1000)

            data = rand(RGBA{Float16}, 100, 100)
            usage = Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_SAMPLED_BIT
            img1 = image(device, data, Vk.FORMAT_R16G16B16A16_SFLOAT; memory_domain = MEMORY_DOMAIN_HOST, optimal_tiling = false, usage)
            @test collect(RGBA{Float16}, img1, device) == data
            img2 = image(device, data, Vk.FORMAT_R16G16B16A16_SFLOAT; memory_domain = MEMORY_DOMAIN_HOST, usage)
            @test collect(RGBA{Float16}, img2, device) == data
            img3 = image(device, data, Vk.FORMAT_R16G16B16A16_SFLOAT; optimal_tiling = false, usage)
            @test collect(RGBA{Float16}, img3, device) == data
            img4 = image(device, data, Vk.FORMAT_R16G16B16A16_SFLOAT; usage)
            @test collect(RGBA{Float16}, img4, device) == data
        end
    end

    @testset "Images" begin
        img = ImageBlock(device, (512, 512), Vk.FORMAT_R32G32B32A32_SFLOAT, Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        @test !isallocated(img)
        @test !isallocated(similar(img))
        allocate!(img, MEMORY_DOMAIN_DEVICE)
        @test isallocated(img)
        @test isallocated(similar(img))
        @test memory(img) isa MemoryBlock
        v = View(img)
        @test v isa ImageView
    end

    @testset "Shaders" begin
        include("shaders.jl")
    end

    @testset "Frame Graph" begin
        include("frame_graph.jl")
    end
end

# trigger finalizers
GC.gc()
