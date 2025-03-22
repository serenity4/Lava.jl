using Graphs: nv, ne

@testset "Render graph" begin
  rg = RenderGraph(device)

  # Specify resources used for the rendering process.
  # Resources can either be physical (e.g. be allocated from a `device` and be bound to GPU memory)
  # or logical (the resource will be created lazily, enabling optimizations such as aliasing or
  # non-allocation of unused resources).

  indirect = buffer_resource(device, rand(100); usage_flags = Vk.BUFFER_USAGE_INDIRECT_BUFFER_BIT)
  lights = buffer_resource(1024)
  emissive = attachment_resource(RGBA{Float32})
  albedo = attachment_resource(RGBA{Float32})
  normal = attachment_resource(RGBA{Float32})
  pbr = attachment_resource(RGBA{Float32})
  color = attachment_resource(RGBA{Float32})
  average_luminance = image_resource(Vk.FORMAT_R32_SFLOAT, [16, 16])
  depth = attachment_resource(Vk.FORMAT_D32_SFLOAT)
  postprocessed = attachment_resource(RGBA{Float32})
  bloom_downsample = image_resource(RGBA{Float32}, [16, 16])

  @test_throws "negative value" RenderArea(-1, 0)
  @test_throws "null value" RenderArea(0, 9)
  @test_throws "pipeline stage must be provided" RenderNode(stages = Vk.PIPELINE_STAGE_2_NONE)
  @test_throws "fragment shader stage must be set" RenderNode(render_area = RenderArea(16, 9), stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT)
  @test_throws "render area must be set" RenderNode(stages = stages = Vk.PIPELINE_STAGE_2_ALL_GRAPHICS_BIT)
  @test_throws "color attachments must be present" RenderNode(fake_graphics_command())
  @test_throws "color attachment must have explicit dimensions" RenderNode(fake_graphics_command(RenderTargets(color)))

  # No rendering done here, doesn't matter what function is passed. We pass in `identity`.
  gbuffer = RenderNode(
    render_area = RenderArea(16, 9),
    stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT | Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
  )
  compute_luminance = RenderNode(stages = Vk.PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
  lighting = RenderNode(render_area = RenderArea(16, 9), stages = Vk.PIPELINE_STAGE_2_ALL_GRAPHICS_BIT)
  postprocess = RenderNode(render_area = RenderArea(16, 9), stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT | Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)

  push!(gbuffer.commands, fake_graphics_command(RenderTargets(emissive, albedo, normal, pbr; depth), draw = DrawIndexedIndirect(indirect, 10)))
  push!(compute_luminance.commands, fake_compute_command())
  push!(lighting.commands, fake_graphics_command(RenderTargets(color, emissive, albedo, normal, pbr; depth)))
  push!(postprocess.commands, fake_graphics_command(RenderTargets(postprocessed, color)))

  @add_resource_dependencies rg begin
    (emissive => (0.0, 0.0, 1.0, 1.0))::Color, albedo::Color, normal::Color, pbr::Color, depth::Depth = gbuffer(indirect::Buffer::Indirect)
    average_luminance::Image::Storage = compute_luminance(lights::Buffer::Physical)
    color::Color = lighting(average_luminance::Image::Storage, emissive::Color, albedo::Color, normal::Color, pbr::Color, depth::Depth)
    postprocessed::Color = postprocess(color::Color, (bloom_downsample * 4)::Texture)
  end

  @testset "Building a render graph" begin
    @test nv(rg.resource_graph) == 4 + 11 # 4 nodes + 10 logical resources + 1 physical resource
    @test ne(rg.resource_graph) == 6 + 2 + 7 + 3

    # Per-node resource usage.
    (; usage) = only(rg.uses[postprocess.id][bloom_downsample.id])
    @test usage.type == RESOURCE_USAGE_TEXTURE
    @test usage.samples == 4

    (; usage) = only(rg.uses[gbuffer.id][emissive.id])
    @test usage.type == RESOURCE_USAGE_COLOR_ATTACHMENT
    @test usage.clear_value == ClearValue((0.0f0, 0.0f0, 1.0f0, 1.0f0))
  end

  @testset "Baking a render graph" begin
    bake!(rg)

    # Combined resource usage.
    (; usage) = rg.combined_resource_uses[color.id]
    @test usage.type == RESOURCE_USAGE_COLOR_ATTACHMENT
    @test usage.access == WRITE | READ
    @test usage.stages == Vk.PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
    @test usage.aspect == Vk.IMAGE_ASPECT_COLOR_BIT
    @test usage.samples == 1
    (; usage) = rg.combined_resource_uses[depth.id]
    @test usage.type == RESOURCE_USAGE_DEPTH_ATTACHMENT
    @test usage.aspect == Vk.IMAGE_ASPECT_DEPTH_BIT
    @test usage.stages == Vk.PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | Vk.PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT

    @test length(rg.materialized_resources) == 10
    @test rg.materialized_resources[color.id].attachment.view.image.usage_flags == Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT
    @test rg.materialized_resources[depth.id].attachment.view.image.usage_flags == Vk.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
    @test rg.materialized_resources[emissive.id].attachment.view.image.usage_flags == Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT

    info = Lava.rendering_info(rg, postprocess)
    @test info.render_area == Vk.Rect2D(Vk.Offset2D(0, 0), Vk.Extent2D(16, 9))
    color_info, postprocessed_info = info.color_attachments
    @test postprocessed_info.image_layout == Vk.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    @test postprocessed_info.load_op == Vk.ATTACHMENT_LOAD_OP_LOAD
    @test postprocessed_info.store_op == Vk.ATTACHMENT_STORE_OP_STORE
    @test convert(Ptr{Cvoid}, postprocessed_info.resolve_image_view) == C_NULL

    @test color_info.image_layout == Vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    @test color_info.load_op == Vk.ATTACHMENT_LOAD_OP_LOAD
    @test color_info.store_op == Vk.ATTACHMENT_STORE_OP_DONT_CARE
    @test convert(Ptr{Cvoid}, color_info.resolve_image_view) == C_NULL

    state = Lava.SynchronizationState()
    info = Lava.dependency_info!(state, rg, gbuffer)
    @test length(info.buffer_memory_barriers) == 1
    @test length(info.image_memory_barriers) == 5
    @test all(barrier.old_layout ≠ barrier.new_layout for barrier in info.image_memory_barriers)

    info = Lava.dependency_info!(state, rg, lighting)
    @test length(info.buffer_memory_barriers) == 0
    @test length(info.image_memory_barriers) == 7

    finish!(rg)

    @testset "Incompatible physical resources" begin
      function bake_with_color(color)
        rg = RenderGraph(device)
        gbuffer_command = fake_graphics_command(RenderTargets(color), draw = DrawIndexedIndirect(indirect, 10))
        gbuffer = RenderNode(gbuffer_command)
        @add_resource_dependencies rg begin
          color::Color = gbuffer(indirect::Buffer::Indirect)
          average_luminance::Image::Storage = compute_luminance(lights::Buffer::Physical)
        end
        bake!(rg)
      end
      color = attachment_resource(device, nothing; format = RGBA{Float32}, dims = [16, 9])
      @test_throws "was provided, but a usage of" bake_with_color(color)
      color = attachment_resource(device, nothing; format = Float32, dims = [16, 9], aspect = Vk.IMAGE_ASPECT_DEPTH_BIT)
      @test_throws "was provided, but is used with an aspect of" bake_with_color(color)
    end
  end

  include("node_synchronization.jl")

  function test_program_vert(position, index, data_address::DeviceAddressBlock)
    pos = @load data_address[index + 1]::Vec2
    position[] = Vec(pos.x, pos.y, 0F, 1F)
  end

  function test_program_frag(out_color)
    out_color[] = Vec(1F, 0F, 0F, 0F)
  end

  function test_program(device)
    vert_shader = @vertex device test_program_vert(::Mutable{Vec4}::Output{Position}, ::UInt32::Input{VertexIndex}, ::DeviceAddressBlock::PushConstant)
    frag_shader = @fragment device test_program_frag(::Mutable{Vec4}::Output)
    Program(vert_shader, frag_shader)
  end

  prog = test_program(device)

  @testset "Rendering" begin
    rg = RenderGraph(device)

    color = attachment_resource(RGBA{Float32})
    normal = image_resource(RGBA{Float32}, [16, 16])
    depth = attachment_resource(Vk.FORMAT_D32_SFLOAT)
    graphics = RenderNode(render_area = RenderArea(16, 9), stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT | Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)

    data = @invocation_data prog @block [Vec2(point...) for point in PointSet(HyperCube(1.0f0), Vec2)]
    push!(graphics.commands, graphics_command(DrawIndexed(collect(1:4)), prog, data, color; depth))

    @add_resource_dependencies rg begin
      (color => (0.0, 0.0, 0.0, 1.0))::Color, depth::Depth = graphics(normal::Texture)
    end

    bake!(rg)
    dependency_info = nothing
    let rg = @set rg.materialized_resources = deepcopy(rg.materialized_resources)
      dependency_info = dependency_info!(SynchronizationState(), rg, graphics)
    end
    rinfo = rendering_info(rg, graphics)
    color_info = Vk.RenderingAttachmentInfo(C_NULL, rg.materialized_resources[color.id].data.view, Vk.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, Vk.RESOLVE_MODE_NONE, C_NULL, Vk.IMAGE_LAYOUT_UNDEFINED, Vk.ATTACHMENT_LOAD_OP_CLEAR, Vk.ATTACHMENT_STORE_OP_STORE, Vk.ClearValue(ClearValue((0f0, 0f0, 0f0, 1f0))))
    depth_info = Vk.RenderingAttachmentInfo(C_NULL, rg.materialized_resources[depth.id].data.view, Vk.IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, Vk.RESOLVE_MODE_NONE, C_NULL, Vk.IMAGE_LAYOUT_UNDEFINED, Vk.ATTACHMENT_LOAD_OP_LOAD, Vk.ATTACHMENT_STORE_OP_STORE, Vk.ClearValue(DEFAULT_CLEAR_VALUE))
    @test rinfo == Vk.RenderingInfo(C_NULL, 0, graphics.render_area.rect, 1, 0, [color_info], depth_info, C_NULL)

    @testset "Barriers for layout transitions" begin
      normal_barrier = Vk.ImageMemoryBarrier2(C_NULL, 0, 0, graphics.stages, Vk.ACCESS_2_SHADER_READ_BIT, Vk.IMAGE_LAYOUT_UNDEFINED, Vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 0, rg.materialized_resources[normal.id].data, Vk.ImageSubresourceRange(Vk.IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1))
      color_barrier = Vk.ImageMemoryBarrier2(C_NULL, 0, 0, Vk.PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, Vk.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, Vk.IMAGE_LAYOUT_UNDEFINED, Vk.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, 0, rg.materialized_resources[color.id].data.view.image, Vk.ImageSubresourceRange(Vk.IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1))
      depth_barrier = Vk.ImageMemoryBarrier2(C_NULL, 0, 0, Vk.PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | Vk.PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, Vk.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, Vk.IMAGE_LAYOUT_UNDEFINED, Vk.IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, 0, 0, rg.materialized_resources[depth.id].data.view.image, Vk.ImageSubresourceRange(Vk.IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1))
      @test dependency_info.image_memory_barriers[1] == normal_barrier
      @test dependency_info.image_memory_barriers[2] == color_barrier
      @test dependency_info.image_memory_barriers[3] == depth_barrier
      @test dependency_info == Vk.DependencyInfo(C_NULL, Vk.DependencyFlag(0), [], [], [normal_barrier, color_barrier, depth_barrier])
    end

    @testset "Recording commands" begin
      empty!(device.pipeline_ht_graphics)
      empty!(device.pending_pipelines_graphics)
      @test isempty(device.pipeline_ht_graphics)
      records, pipeline_hashes = Lava.record_commands!(rg)
      @test isempty(device.pipeline_ht_graphics)
      Lava.create_pipelines!(device)
      @test !isempty(device.pipeline_ht_graphics)
      pipeline = device.pipeline_ht_graphics[only(pipeline_hashes)]

      command_buffer = Lava.SnoopCommandBuffer()
      Lava.fill_indices!(rg.index_data, records)
      Lava.initialize_index_buffer(command_buffer, device, rg.index_data)
      flush(command_buffer, rg, records, pipeline_hashes)
      @test !isempty(command_buffer)
      @test getproperty.(command_buffer.records, :name) == [:cmd_bind_index_buffer, :cmd_pipeline_barrier_2, :cmd_begin_rendering, :cmd_bind_pipeline, :cmd_bind_descriptor_sets, :cmd_push_constants, :cmd_set_depth_test_enable, :cmd_set_depth_write_enable, :cmd_set_depth_compare_op, :cmd_set_stencil_test_enable, :cmd_set_stencil_op, :cmd_set_stencil_compare_mask, :cmd_set_stencil_write_mask, :cmd_set_stencil_reference, :cmd_set_stencil_op, :cmd_set_stencil_compare_mask, :cmd_set_stencil_write_mask, :cmd_set_stencil_reference, :cmd_draw_indexed, :cmd_end_rendering]
      _cmd_bind_index_buffer, _cmd_pipeline_barrier_2, _cmd_begin_rendering, _cmd_bind_pipeline, _cmd_bind_descriptor_sets, _cmd_push_constants, _cmd_set_depth_test_enable, _cmd_set_depth_write_enable, _cmd_set_depth_compare_op, _cmd_set_stencil_test_enable, _cmd_set_stencil_op, _cmd_set_stencil_compare_mask, _cmd_set_stencil_write_mask, _cmd_set_stencil_reference, _cmd_set_stencil_op, _cmd_set_stencil_compare_mask, _cmd_set_stencil_write_mask, _cmd_set_stencil_reference, _cmd_draw_indexed, _cmd_end_rendering = command_buffer

      @test isallocated(_cmd_bind_index_buffer.args[1])
      @test _cmd_bind_index_buffer.args == [rg.index_data.index_buffer[], 0, Vk.INDEX_TYPE_UINT32]

      @test _cmd_pipeline_barrier_2.args == [dependency_info]
      @test _cmd_begin_rendering.args == [rinfo]
      @test _cmd_bind_pipeline.args == [Vk.PIPELINE_BIND_POINT_GRAPHICS, pipeline]
      @test _cmd_bind_descriptor_sets.args == [Vk.PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 0, [device.descriptors.gset], []]
      @test _cmd_push_constants.args[1:2] == [pipeline.layout, Vk.SHADER_STAGE_ALL]
      @test _cmd_push_constants.args[4] == sizeof(DeviceAddressBlock)
      @test _cmd_draw_indexed.args == [4, 1, 0, -1, 0]
      @test _cmd_end_rendering.args == []

      test_validation_msg(x -> @test isempty(x)) do
        command_buffer = Lava.request_command_buffer(device)
        Lava.fill_indices!(rg.index_data, records)
        Lava.initialize_index_buffer(command_buffer, device, rg.index_data)
        flush(command_buffer, rg, records, pipeline_hashes)
      end

      finish!(rg)
    end
  end

  @testset "Render graph from persistent data" begin
    color = attachment_resource(RGBA{Float32})
    depth = attachment_resource(Vk.FORMAT_D32_SFLOAT)
    graphics = RenderNode(render_area = RenderArea(16, 9), stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT | Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)
    draw = DrawIndexed(1:4)
    dependencies = @resource_dependencies begin
      @write
      (color => (0.0, 0.0, 0.0, 1.0))::Color
      depth::Depth
    end
    command = graphics_command(draw, prog, @invocation_data(prog, @block collect(PointSet(HyperCube(1.0f0), Vec2))), RenderTargets(color; depth), RenderState(), ProgramInvocationState(), dependencies)
    push!(graphics.commands, command)

    rg = RenderGraph(device)
    add_node!(rg, graphics)
    @test collect(rg.nodes) == [graphics]
    bake!(rg)
    @test command.graphics.data_address ≠ DeviceAddressBlock(0)

    # Make sure debug logging works correctly.
    mktemp() do path, io
      withenv("JULIA_DEBUG" => "Lava") do
        redirect_stdio(stdout=io, stderr=io) do
          @test render(device, graphics)
        end
        seekstart(io)
        @test !isempty(read(io, String))
      end
    end
    finish!(rg)
  end
end;
