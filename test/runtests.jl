is_ci = get(ENV, "JULIA_CI", "false") == "true"
if is_ci
  import SwiftShader_jll
  ENV["JULIA_VULKAN_LIBNAME"] = basename(SwiftShader_jll.libvulkan)
end

using Lava, Dictionaries
using SPIRV: SPIRV, Pointer, ShaderInterface, U, F

is_ci && Vk.@set_driver :SwiftShader
using Test
using GeometryExperiments
using FixedPointNumbers
using FileIO
using ImageIO
using Accessors
using OpenType
# XCB must be loaded prior to creating the instance that will use VK_KHR_xcb_surface.
using XCB: XCB, Connection, current_screen, XCBWindow, XWindowManager

include("utils.jl")
instance, device = init(; with_validation = !is_ci, instance_extensions = ["VK_KHR_xcb_surface"])

@testset "Lava.jl" begin
  @testset "Initialization" begin
    @testset "SPIR-V capability/extension detection" begin
      (; physical_device) = device.handle
      feats = Lava.spirv_features(physical_device, device.api_version, device.extensions, device.features)
      @test !isempty(feats.extensions)
      @test !isempty(feats.capabilities)
      @test SPIRV.CapabilityVulkanMemoryModel in feats.capabilities
      @test SPIRV.CapabilityShader in feats.capabilities
      feats = Lava.spirv_features(physical_device, device.api_version, [], Vk.initialize(Vk.PhysicalDeviceFeatures2))
      @test SPIRV.CapabilityVulkanMemoryModel ∉ feats.capabilities
      @test SPIRV.CapabilityShader in feats.capabilities
    end
  end

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
    yield()
    test_validation_msg(x -> @test startswith(x, "┌ Error: Validation")) do
      too_much = Lava.memory_block(device, 100000000000000000, 7, MEMORY_DOMAIN_DEVICE)
      @test iserror(too_much)
      @test unwrap_error(too_much).code == Vk.ERROR_OUT_OF_DEVICE_MEMORY
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

    @test buffer(device, collect(1:1000); memory_domain = MEMORY_DOMAIN_HOST) isa Buffer
    @test buffer(device, collect(1:1000)) isa Buffer
    @test buffer(device; size = 800) isa Buffer
    @test_throws ErrorException buffer(device)

    @testset "Allocators" begin
      la = LinearAllocator(device, 1000)
      @test size(la) == available_size(la) == 1000
      @test device_address(la) ≠ C_NULL

      sub = copyto!(la, [1, 2, 3])
      @test offset(sub) == 0
      @test size(sub) == 24
      @test available_size(la) == size(la) - size(sub) == 976
      @test available_size(la, 16) == 968
      sub = copyto!(la, (4.0f0, 5.0f0, 6.0f0))
      @test offset(sub) == 24
      @test size(sub) == 12
      sub = copyto!(la, (4.0f0, 5.0f0, 6.0f0))
      # 8-byte alignment requirement
      @test offset(sub) == 40

      Lava.reset!(la)
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

      img = image(device; format = Vk.FORMAT_R32G32B32A32_SFLOAT, dims = (512, 512))
      @test img isa Lava.Image
      @test eltype(img) == RGBA{Float32}
      img = image(device, rand(RGBA{Float32}, 512, 512))
      @test isallocated(img)
    end

    @testset "Data transfer" begin
      b1 = buffer(device, collect(1:1000); usage = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT, memory_domain = MEMORY_DOMAIN_HOST)
      b2 = buffer(device; size = 8000, usage = Vk.BUFFER_USAGE_TRANSFER_DST_BIT | Vk.BUFFER_USAGE_TRANSFER_SRC_BIT)
      @test reinterpret(Int64, collect(b1)) == collect(1:1000)
      transfer(device, b1, b2; submission = sync_submission(device))
      @test reinterpret(Int64, collect(b2, device)) == collect(1:1000)

      b3 = buffer(device, collect(1:1000); usage = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT)
      @test reinterpret(Int64, collect(b3, device)) == collect(1:1000)

      data = rand(RGBA{Float16}, 100, 100)
      usage = Vk.IMAGE_USAGE_TRANSFER_SRC_BIT
      img1 = image(device, data; memory_domain = MEMORY_DOMAIN_HOST, optimal_tiling = false, usage)
      @test collect(img1, device) == data
      img2 = image(device, data; memory_domain = MEMORY_DOMAIN_HOST, usage)
      @test collect(img2, device) == data
      img3 = image(device, data; optimal_tiling = false, usage)
      @test collect(img3, device) == data
      img4 = image(device, data; usage)
      @test collect(img4, device) == data
    end
  end

  include("resources.jl")
  include("fence_pool.jl")
  include("descriptors.jl")
  include("data_blocks.jl")
  include("shaders.jl")
  include("simple_program.jl")
  include("pipelines.jl")
  include("render_graph.jl")
  include("examples.jl")
  include("cycles.jl")
  include("present.jl")

  # Make sure we don't have fences that are never signaled.
  Lava.compact!(device.fence_pool)
  @test isempty(device.fence_pool.pending)
  @test isempty(device.fence_pool.completed)
  @test !isempty(device.fence_pool.available)
end;

# trigger finalizers
GC.gc()
