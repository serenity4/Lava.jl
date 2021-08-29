using Lava
using Test

@testset "Lava.jl" begin
    instance, device = init()
    buffer = BufferBlock(device, 100, Vk.BUFFER_USAGE_TRANSFER_DST_BIT)
    @test !isallocated(buffer)
    subbuffer = @view buffer[2:4:end]
    @test subbuffer isa SubBuffer
    @test_throws UndefRefError memory(subbuffer)

    mem = unwrap(MemoryBlock(device, 100, 7, MEMORY_DOMAIN_HOST_CACHED))
    submem = @view mem[2:4]
    @test submem isa SubMemory

    unwrap(allocate!(buffer, MEMORY_DOMAIN_HOST_CACHED))
    @test isallocated(buffer)
    @test memory(buffer) isa MemoryBlock
    @test memory(subbuffer) isa SubMemory

    image = ImageBlock(device, (512, 512), Vk.FORMAT_R32G32B32A32_SFLOAT, Vk.IMAGE_USAGE_TRANSFER_DST_BIT)
    @test !isallocated(image)
    allocate!(image, MEMORY_DOMAIN_DEVICE)
    @test isallocated(image)
    @test memory(image) isa MemoryBlock
end
