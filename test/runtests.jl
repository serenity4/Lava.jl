using Lava
using Lava: Memory
using Test
using Random: Random, MersenneTwister, AbstractRNG
using Dictionaries
using SPIRV: SPIRV, @compile, validate, @for
using Swizzles: @swizzle
using SPIRV.MathFunctions
using SymbolicGA: @ga
using GeometryExperiments: GeometryExperiments, Mesh, VertexMesh, subdivide!, UniformSubdivision, triangulate!, orientation, FACE_ORIENTATION_COUNTERCLOCKWISE, BezierCurve, Box, PointSet, HyperCube, MeshEncoding, reencode, MESH_TOPOLOGY_TRIANGLE_LIST
using ColorTypes: AbstractRGBA
using FixedPointNumbers
using FileIO, ImageIO
using VideoIO
using Accessors
# XCB must be loaded prior to creating the instance that will use VK_KHR_xcb_surface.
using XCB: XCB, XWindowManager, current_screen, XCBWindow, resize, extent
using ImageMagick: clamp01nan, clamp01nan!
using Distances: Distances, PeriodicEuclidean
using OpenType: curves, curves_normalized, Text, Line
using OpenType

using Lava: request_index!, GlobalDescriptors, DescriptorArray, patch_descriptors!, patch_pointers!, device_address_block!, RESOURCE_TYPE_IMAGE, RESOURCE_TYPE_BUFFER, RESOURCE_TYPE_ATTACHMENT, assert_type, resource_type, descriptor_type, islogical, isphysical, DESCRIPTOR_TYPE_TEXTURE, get_descriptor_index!, delete_descriptor!, NodeID, free_descriptor_batch!, fence_status, compact!, FencePool, request_command_buffer, ShaderCache, combine_resource_uses_per_node, combine_resource_uses, isbuffer, isimage, isattachment, SynchronizationState, bake!, dependency_info!, rendering_info, PROGRAM_TYPE_GRAPHICS, PROGRAM_TYPE_COMPUTE, COMMAND_TYPE_DRAW_INDEXED, COMMAND_TYPE_DRAW_INDEXED_INDIRECT, Image

include("utils.jl")
instance, device = init(; with_validation = true, instance_extensions = ["VK_KHR_xcb_surface"])

@testset "Lava.jl" begin
  include("subresources.jl")

  @testset "Buffers & Memory" begin
    b = Buffer(device, 100)
    @test_throws UndefRefError b.memory[]
    @test !isallocated(b)

    sb = similar(b)
    @test sb.size == b.size
    @test sb.sharing_mode == b.sharing_mode
    @test sb.queue_family_indices == b.queue_family_indices
    @test sb.usage_flags == b.usage_flags
    @test !isallocated(sb)
    @test sb.memory ≠ b.memory

    sub = @view b[2:4:end]
    @test sub.offset == 2
    @test sub.stride == 4
    @test sub.size == 96

    sub = @view b[2:end]
    @test sub.offset == 2
    @test sub.stride == 0
    @test sub.size == b.size - 2

    mem = Memory(device, 100, 7, MEMORY_DOMAIN_HOST_CACHED)
    submem = @view mem[2:5]
    @test submem.offset == 2
    @test submem.size == 3
    yield()
    test_validation_msg(x -> @test startswith(x, "┌ Error: Validation")) do
      too_much = Lava.allocate_memory(device, 100000000000000000, 7, MEMORY_DOMAIN_DEVICE)
      @test iserror(too_much)
      @test unwrap_error(too_much).code == Vk.ERROR_OUT_OF_DEVICE_MEMORY

      @test_throws Lava.OutOfDeviceMemoryError(100000000000000000) Memory(device, 100000000000000000, 7, MEMORY_DOMAIN_DEVICE)
      @test_throws Lava.OutOfDeviceMemoryError(100000000000000000) Memory(device, 100000000000000000, 7, MEMORY_DOMAIN_HOST)
    end

    allocate!(b, MEMORY_DOMAIN_HOST_CACHED)
    @test isallocated(b)
    @test DeviceAddress(b) ≠ DeviceAddress(C_NULL)
    @test DeviceAddress(sub) == DeviceAddress(DeviceAddress(b) + sub.offset)
    mem2 = Memory(device, 1000, 7, MEMORY_DOMAIN_HOST_CACHED)
    b2 = Buffer(device, 100)
    bind!(b2, mem2)
    @test b2.memory[] === mem2

    sb = similar(b, memory_domain = MEMORY_DOMAIN_DEVICE)
    @test isallocated(sb)
    @test sb.memory[] ≠ b.memory[]

    @test isallocated(Buffer(device; data = collect(1:1000), memory_domain = MEMORY_DOMAIN_HOST))
    @test isallocated(Buffer(device; data = collect(1:1000)))
    @test isallocated(Buffer(device; size = 800))
    @test_throws "must be provided" Buffer(device)

    @testset "Allocators" begin
      la = LinearAllocator(device, 1000)
      @test la.buffer.size == available_size(la) == 1000
      @test DeviceAddress(la) ≠ DeviceAddress(C_NULL)

      sub = copyto!(la, [1, 2, 3])
      @test sub.offset == 0
      @test sub.size == 24
      @test available_size(la) == la.buffer.size - sub.size == 976
      @test available_size(la, 16) == 968
      sub = copyto!(la, (4.0f0, 5.0f0, 6.0f0))
      @test sub.offset == 24
      @test sub.size == 12
      sub = copyto!(la, (4.0f0, 5.0f0, 6.0f0))
      # 8-byte alignment requirement
      @test sub.offset == 36

      Lava.reset!(la)
      @test la.last_offset == 0
    end

    @testset "Images" begin
      img = Image(device, [512, 512], RGBA{Float32}, Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
      @test !isallocated(img)
      @test !isallocated(similar(img))
      allocate!(img, MEMORY_DOMAIN_DEVICE)
      @test isallocated(img)
      @test isallocated(similar(img))

      v = ImageView(img)
      @test isa(v, ImageView)

      img = Image(device; format = RGBA{Float32}, dims = [512, 512])
      @test img isa Lava.Image
      @test eltype(img) == RGBA{Float32}
      img = Image(device; data = rand(RGBA{Float32}, 512, 512))
      @test isallocated(img)
    end

    @testset "Data transfer" begin
      b1 = Buffer(device; data = collect(1:1000), usage_flags = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT, memory_domain = MEMORY_DOMAIN_HOST)
      b2 = Buffer(device; size = 8000, usage_flags = Vk.BUFFER_USAGE_TRANSFER_DST_BIT | Vk.BUFFER_USAGE_TRANSFER_SRC_BIT)
      @test reinterpret(Int64, collect(b1)) == collect(1:1000)
      @test collect(Int64, b1) == collect(1:1000)
      transfer(device, b1, b2; submission = sync_submission(device))
      @test reinterpret(Int64, collect(b2, device)) == collect(1:1000)
      @test collect(Int64, b2, device) == collect(1:1000)

      b3 = Buffer(device; data = collect(1:1000), usage_flags = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT)
      @test reinterpret(Int64, collect(b3, device)) == collect(1:1000)

      # Include the sampled bit to avoid validation layers complaining for image views.
      usage_flags = Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_SAMPLED_BIT

      data = rand(RGBA{Float16}, 200, 100)
      img1 = Image(device; data, memory_domain = MEMORY_DOMAIN_HOST, optimal_tiling = false, usage_flags)
      @test collect(img1, device) == data
      img2 = Image(device; data, memory_domain = MEMORY_DOMAIN_HOST, optimal_tiling = false, usage_flags)
      @test collect(img2, device) == data
      img3 = Image(device; data, memory_domain = MEMORY_DOMAIN_DEVICE, optimal_tiling = false, usage_flags)
      @test collect(img3, device) == data
      img4 = Image(device; data, memory_domain = MEMORY_DOMAIN_DEVICE, optimal_tiling = true, usage_flags)
      @test collect(img4, device) == data

      data = [rand(RGBA{Float16}, 256, 256) for _ in 1:6]
      flags = Vk.IMAGE_CREATE_CUBE_COMPATIBLE_BIT
      cubemap = Image(device; data, layers = 6, usage_flags, flags)
      for i in 1:6
        face = ImageView(cubemap; layer_range = i:i)
        @test collect(face, device) == data[i]
      end

      image = Image(device; format = RGBA{Float16}, dims = [256, 256], layers = 6, mip_levels = 4, usage_flags = usage_flags | Vk.IMAGE_USAGE_TRANSFER_DST_BIT)
      for layer in 1:6
        for mip_level in 1:4
          layer = 1
          mip_level = 1
          let view = ImageView(image; layer_range = layer:layer, mip_range = mip_level:mip_level)
            n = 256 >> (mip_level - 1)
            data = rand(RGBA{Float16}, n, n)
            copyto!(view, data, device)
            @test collect(view, device) == data
          end
        end
      end
    end
  end

  include("resources.jl")
  include("fence_pool.jl")
  include("descriptors.jl")
  include("data_blocks.jl")
  include("shaders.jl")
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
