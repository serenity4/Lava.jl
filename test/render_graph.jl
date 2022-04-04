using Lava, Accessors, Dictionaries, GeometryExperiments
using Test
using Graphs: nv, ne
instance, device = init(; with_validation = true, device_specific_features = [:shader_int_64, :sampler_anisotropy])

rg = RenderGraph(device)

# Specify resources used for the rendering process.
# Resources can either be physical (e.g. be allocated from a `device` and be bound to GPU memory)
# or logical (the resource will be created lazily, enabling optimizations such as aliasing or
# non-allocation of unused resources).

# Logical resource.
vbuffer = buffer(rg, 1024)
@test vbuffer isa LogicalBuffer
# Regular `BufferBlock`.
ibuffer = wait(buffer(device, collect(1:100); usage = Vk.BUFFER_USAGE_INDEX_BUFFER_BIT))
@test ibuffer isa BufferBlock

# Logical resources all the way.
average_luminance = image(rg, Vk.FORMAT_R32G32B32A32_SFLOAT, (16, 16))
emissive = attachment(rg, Vk.FORMAT_R32G32B32A32_SFLOAT)
albedo = attachment(rg, Vk.FORMAT_R32G32B32A32_SFLOAT)
normal = attachment(rg, Vk.FORMAT_R32G32B32A32_SFLOAT)
pbr = attachment(rg, Vk.FORMAT_R32G32B32A32_SFLOAT)
color = attachment(rg, Vk.FORMAT_R32G32B32A32_SFLOAT; samples = 4)
output = attachment(rg, Vk.FORMAT_R32G32B32A32_SFLOAT)
depth = attachment(rg, Vk.FORMAT_D32_SFLOAT)
shadow_main = image(rg, Vk.FORMAT_D32_SFLOAT, (16, 16))
shadow_near = image(rg, Vk.FORMAT_D32_SFLOAT, (16, 16))
bloom_downsample_3 = image(rg, Vk.FORMAT_R32G32B32A32_SFLOAT, (16, 16))

# No rendering done here, doesn't matter what function is passed. We pass in `identity`.
gbuffer = RenderNode(
  identity;
  render_area = RenderArea(1920, 1080),
  stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT | Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
)
lighting = RenderNode(identity; render_area = RenderArea(1920, 1080), stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT)
adapt_luminance = RenderNode(identity, render_area = RenderArea(1920, 1080), stages = Vk.PIPELINE_STAGE_2_ALL_GRAPHICS_BIT)
combine = RenderNode(identity, render_area = RenderArea(1920, 1080), stages = Vk.PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)

@add_resource_dependencies rg begin
  (emissive => (0.0, 0.0, 0.0, 1.0))::Color, albedo::Color, normal::Color, pbr::Color, depth::Depth =
    gbuffer(vbuffer::Buffer::Vertex, ibuffer::Buffer::Index)
  color::Color = lighting(emissive::Color, albedo::Color, normal::Color, pbr::Color, depth::Depth, shadow_main::Texture, shadow_near::Texture)
  average_luminance::Image::Storage = adapt_luminance(average_luminance::Image::Storage, (bloom_downsample_3 * 4)::Texture)
  output::Color = combine(color::Color, average_luminance::Texture)
end

@testset "Building a render graph" begin
  @test nv(rg.resource_graph) == 4 + 13
  @test ne(rg.resource_graph) == 6 + 8 + 3 + 3

  # Per-node resource usage.
  usage = rg.uses[adapt_luminance.uuid][bloom_downsample_3]
  @test usage.type == RESOURCE_TYPE_TEXTURE
  @test usage.samples == 4

  usage = rg.uses[gbuffer.uuid][emissive]
  @test usage.type == RESOURCE_TYPE_COLOR_ATTACHMENT
  @test usage.clear_value == (0.0f0, 0.0f0, 0.0f0, 1.0f0)

  # Combined resource usage.
  uses = Lava.ResourceUses(rg)

  usage = uses[color]
  @test usage.type == RESOURCE_TYPE_COLOR_ATTACHMENT
  @test usage.access == WRITE | READ
  @test usage.stages == Vk.PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT
  @test usage.aspect == Vk.IMAGE_ASPECT_COLOR_BIT
  @test usage.samples == 1

  usage = uses[depth]
  @test usage.type == RESOURCE_TYPE_DEPTH_ATTACHMENT
  @test usage.aspect == Vk.IMAGE_ASPECT_DEPTH_BIT

  @test isnothing(Lava.check_physical_resources(rg, uses))

  uses2 = deepcopy(uses)
  ibuffer_uuid = findfirst(x -> Vk.BUFFER_USAGE_INDEX_BUFFER_BIT in x.usage, rg.physical_resources.buffers)
  ibuffer_usage = uses2[ibuffer_uuid]
  set!(uses2.buffers, ibuffer_uuid, @set ibuffer_usage.usage = Vk.BUFFER_USAGE_STORAGE_BUFFER_BIT)
  @test_throws ErrorException Lava.check_physical_resources(rg, uses2)

  resources = Lava.materialize_logical_resources(rg, uses)
  @test resources isa Lava.PhysicalResources
  @test resources[color].usage == Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT
end

@testset "Baking a render graph" begin
  baked = Lava.bake(rg)
  info = Lava.rendering_info(baked, combine)
  @test info.render_area == Vk.Rect2D(Vk.Offset2D(0, 0), Vk.Extent2D(1920, 1080))
  color_info, output_info = info.color_attachments
  @test output_info.image_layout == Vk.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
  @test output_info.load_op == Vk.ATTACHMENT_LOAD_OP_LOAD
  @test output_info.store_op == Vk.ATTACHMENT_STORE_OP_STORE
  @test convert(Ptr{Cvoid}, output_info.resolve_image_view) == C_NULL

  @test color_info.image_layout == Vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
  @test color_info.load_op == Vk.ATTACHMENT_LOAD_OP_LOAD
  @test color_info.store_op == Vk.ATTACHMENT_STORE_OP_DONT_CARE
  @test convert(Ptr{Cvoid}, color_info.resolve_image_view) == C_NULL

  state = Lava.SynchronizationState()
  info = Lava.dependency_info!(state, baked, gbuffer)
  # No synchronization required for first use.
  @test length(info.buffer_memory_barriers) == 0
  # Except for image layout transitions.
  @test length(info.image_memory_barriers) == 5
  @test all(barrier.old_layout ≠ barrier.new_layout for barrier in info.image_memory_barriers)

  info = Lava.dependency_info!(state, baked, lighting)
  @test length(info.buffer_memory_barriers) == 0
  @test length(info.image_memory_barriers) == 8
end

include("simple_program.jl")
prog = simple_program(device)

@testset "Rendering" begin
  rg = RenderGraph(device)

  color = attachment(rg, Vk.FORMAT_R32G32B32A32_SFLOAT)
  normal = image(rg, Vk.FORMAT_R32G32B32A32_SFLOAT, (16, 16))
  depth = attachment(rg, Vk.FORMAT_D32_SFLOAT)
  graphics =
    RenderNode(render_area = RenderArea(1920, 1080), stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT | Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT) do rec
      set_program(rec, prog)
      set_material(rec, index(rec, Texture(rec, normal)))
      draw(rec, PointSet(HyperCube(1.0f0), Point{2,Float32}).points, collect(1:4), color; depth)
    end

  @add_resource_dependencies rg begin
    (color => (0.0, 0.0, 0.0, 1.0))::Color, depth::Depth = graphics(normal::Texture)
  end

  baked = Lava.bake(rg)
  empty!(device.pipeline_ht)
  @test isempty(device.pipeline_ht)
  records, pipeline_hashes = Lava.record_commands!(baked)
  @test isempty(device.pipeline_ht)
  Lava.create_pipelines(device)
  @test !isempty(device.pipeline_ht)
  command_buffer = Lava.request_command_buffer(device)
  # command_buffer = Lava.SnoopCommandBuffer()
  Lava.initialize(command_buffer, device, baked.global_data)
  flush(command_buffer, baked, records, pipeline_hashes)
end
