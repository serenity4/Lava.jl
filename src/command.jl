abstract type Command end

"""
Operation lazily executed, for example recorded in a [`CommandRecord`](@ref).
"""
abstract type LazyOperation <: Command end

"""
Operation that sets rendering state for invoking further operations, but which does not do any work by itself.
"""
abstract type StateCommand <: Command end

"""
Copy operation from one source to a destination.
"""
abstract type Copy{S,D} <: LazyOperation end

"""
Type that records command lazily, for them to be flushed into an Vulkan command buffer later.
"""
abstract type CommandRecord <: LavaAbstraction end

function draw(record::CommandRecord, command::DrawCommand, program::Program, targets::RenderTargets, state::DrawState)
  program_draws = get!(Dictionary, record.programs, program)
  commands = get!(Vector{DrawCommand}, program_draws, state)
  push!(commands, command => targets)
  nothing
end

draw(record::CommandRecord, info::DrawInfo) = draw(record, info.command, info.program, info.targets, info.state)

"""
Record that compacts action commands according to their program and state before flushing.

This allows to group draw calls that use the exact same rendering state for better performance.
"""
struct CompactRecord <: CommandRecord
  node::RenderNode
  programs::Dictionary{Program,Dictionary{DrawState,Vector{Pair{DrawCommand,RenderTargets}}}}
end

function CompactRecord(baked::BakedRenderGraph, node::RenderNode)
  rec = CompactRecord(node, Dictionary())
  for info in node.draw_infos
    draw(rec, @set info.targets = materialize(baked, info.targets))
  end
  rec
end

Base.show(io::IO, record::CompactRecord) = print(
  io,
  CompactRecord,
  '(',
  length(record.programs),
  " programs, $(sum(x -> sum(length, values(x); init = 0), values(record.programs); init = 0)) draw commands)",
)

function DrawInfo(rg::RenderGraph, program::Program, vdata, idata, color...; depth = nothing, stencil = nothing, material = nothing, instances = 1:1, render_state::RenderState = RenderState(), invocation_state::ProgramInvocationState = ProgramInvocationState())
  data = DrawData(;
    vertex_data = allocate_vertex_data(rg, program, vdata),
    material_data = isnothing(material) ? 0 : allocate_material(rg, program, material),
  )
  state = DrawState(render_state, invocation_state, data)
  command = DrawIndexed(0, append!(rg.index_data, idata), instances)
  targets = RenderTargets(color...; depth, stencil)
  DrawInfo(command, targets, state, program)
end

struct DrawDirect <: DrawCommand
  vertices::UnitRange{Int64}
  instances::UnitRange{Int64}
end

function apply(cb::CommandBuffer, draw::DrawDirect)
  Vk.cmd_draw(
    cb,
    1 + draw.vertices.stop - draw.vertices.start,
    1 + draw.instances.stop - draw.instances.start,
    draw.vertices.start - 1,
    draw.instances.start - 1,
  )
end

struct DrawIndirect{B<:Buffer} <: DrawCommand
  parameters::B
  count::Int64
end

function apply(cb::CommandBuffer, draw::DrawIndirect)
  buffer = draw.parameters
  Vk.cmd_draw_indirect(cb, buffer, offset(buffer), draw.count, stride(buffer))
end

struct DrawIndexed <: DrawCommand
  vertex_offset::Int64
  indices::UnitRange{Int64}
  instances::UnitRange{Int64}
end

function apply(cb::CommandBuffer, draw::DrawIndexed)
  Vk.cmd_draw_indexed(
    cb,
    1 + draw.indices.stop - draw.indices.start,
    1 + draw.instances.stop - draw.instances.start,
    draw.indices.start - 1,
    draw.vertex_offset,
    draw.instances.start - 1,
  )
end

struct DrawIndexedIndirect{B<:Buffer} <: DrawCommand
  parameters::B
  count::Int64
end

function apply(cb::CommandBuffer, draw::DrawIndexedIndirect)
  buffer = draw.parameters
  Vk.cmd_draw_indexed_indirect(cb, buffer, offset(buffer), draw.count, stride(buffer))
end

function request_pipelines(baked::BakedRenderGraph, record::CompactRecord)
  pipeline_hashes = Dictionary{ProgramInstance,UInt64}()
  for (program, calls) in pairs(record.programs)
    for (state, draws) in pairs(calls)
      for targets in unique!(last.(draws))
        info = pipeline_info(baked.device, record.node.render_area.rect::Vk.Rect2D, program, state.render_state, state.program_state, baked.descriptors, targets)
        hash = request_pipeline(baked.device, info)
        set!(pipeline_hashes, ProgramInstance(program, state, targets), hash)
      end
    end
  end
  pipeline_hashes
end

function materialize(baked::BakedRenderGraph, targets::RenderTargets)
  color = map(targets.color) do c
    isa(c, PhysicalAttachment) ? c : baked.resources.attachments[uuid(c)]
  end
  depth = isa(targets.depth, LogicalAttachment) ? baked.resources.attachments[uuid(targets.depth)] : targets.depth
  stencil = isa(targets.stencil, LogicalAttachment) ? baked.resources.attachments[uuid(targets.stencil)] : targets.stencil
  RenderTargets(color, depth, stencil)
end

"""
Submit a pipeline create info for creation in the next batch.

A hash is returned to serve as the key to get the corresponding pipeline from the hash table.
"""
function pipeline_info(
  device::Device,
  render_area::Vk.Rect2D,
  program::Program,
  state::RenderState,
  invocation_state::ProgramInvocationState,
  resources::PhysicalDescriptors,
  targets::RenderTargets,
)
  shader_stages = [Vk.PipelineShaderStageCreateInfo(shader) for shader in program.shaders]
  # Vertex data is retrieved from an address provided in the push constant.
  vertex_input_state = Vk.PipelineVertexInputStateCreateInfo([], [])
  rendering_state = Vk.PipelineRenderingCreateInfo(0, format.(targets.color), format(targets.depth), format(targets.stencil))
  attachments = map(targets.color) do _
    if isnothing(state.blending_mode)
      #TODO: Allow specifying blending mode for color attachments.
      Vk.PipelineColorBlendAttachmentState(
        true,
        Vk.BLEND_FACTOR_SRC_ALPHA,
        Vk.BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        Vk.BLEND_OP_ADD,
        Vk.BLEND_FACTOR_SRC_ALPHA,
        Vk.BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        Vk.BLEND_OP_ADD;
        color_write_mask = state.color_write_mask,
      )
    else
      error("Color blending not supported")
    end
  end
  input_assembly_state = Vk.PipelineInputAssemblyStateCreateInfo(invocation_state.primitive_topology, false)
  (; x, y) = render_area.offset
  (; width, height) = render_area.extent
  viewport_state =
    Vk.PipelineViewportStateCreateInfo(
      viewports = [Vk.Viewport(x, height - y, float(width), -float(height), 0, 1)],
      scissors = [render_area],
    )

  (; depth_bias) = state
  use_depth_bias = !isnothing(depth_bias)
  depth_bias_constant_factor = depth_bias_clamp = depth_bias_slope_factor = 0.0f0
  if use_depth_bias
    depth_bias_constant_factor = depth_bias.constant_factor
    depth_bias_clamp = depth_bias.clamp
    depth_bias_slope_factor = depth_bias.slope
  end

  rasterizer = Vk.PipelineRasterizationStateCreateInfo(
    false,
    false,
    invocation_state.polygon_mode,
    invocation_state.triangle_orientation,
    use_depth_bias,
    depth_bias_constant_factor,
    depth_bias_clamp,
    depth_bias_slope_factor,
    1.0,
    cull_mode = invocation_state.cull_mode,
  )
  nsamples = samples(first(targets.color))
  all(==(nsamples) ∘ samples, targets.color) || error("Incoherent number of samples detected: $(samples.(targets.color))")
  multisample_state = Vk.PipelineMultisampleStateCreateInfo(Vk.SampleCountFlag(nsamples), false, 1.0, false, false)
  color_blend_state = Vk.PipelineColorBlendStateCreateInfo(false, Vk.LOGIC_OP_AND, attachments, ntuple(Returns(1.0f0), 4))
  layout = pipeline_layout(device, resources)
  depth_stencil_state = C_NULL
  if !isnothing(targets.depth) || !isnothing(targets.stencil)
    depth_stencil_state = Vk.PipelineDepthStencilStateCreateInfo(
      state.enable_depth_testing,
      true, # depth_write_enable
      Vk.COMPARE_OP_LESS_OR_EQUAL,
      false, # depth_bounds_enable
      false, # stencil test enable
      Vk.StencilOpState(Vk.STENCIL_OP_KEEP, Vk.STENCIL_OP_KEEP, Vk.STENCIL_OP_KEEP, Vk.COMPARE_OP_LESS_OR_EQUAL, 0, 0, 0),
      Vk.StencilOpState(Vk.STENCIL_OP_KEEP, Vk.STENCIL_OP_KEEP, Vk.STENCIL_OP_KEEP, Vk.COMPARE_OP_LESS_OR_EQUAL, 0, 0, 0),
      typemin(Float32),
      typemax(Float32),
    )
  end
  Vk.GraphicsPipelineCreateInfo(
    shader_stages,
    rasterizer,
    handle(layout),
    0,
    0;
    next = rendering_state,
    input_assembly_state,
    vertex_input_state,
    viewport_state,
    multisample_state,
    color_blend_state,
    depth_stencil_state,
  )
end

function request_pipeline(device::Device, info::Vk.GraphicsPipelineCreateInfo)
  push!(device.pending_pipelines, info)
  hash(info)
end

function Base.flush(cb::CommandBuffer, record::CompactRecord, device::Device, binding_state::BindState, pipeline_hashes, descriptors::PhysicalDescriptors)
  for (program, calls) in pairs(record.programs)
    for (state, draws) in pairs(calls)
      for (call, targets) in draws
        hash = pipeline_hashes[ProgramInstance(program, state, targets)]
        pipeline = device.pipeline_ht[hash]
        reqs = BindRequirements(pipeline, state.push_data, descriptors.gset.set)
        bind(cb, reqs, binding_state)
        binding_state = reqs
        apply(cb, call)
      end
    end
  end
  binding_state
end

function initialize(cb::CommandBuffer, device::Device, id::IndexData, descriptors::PhysicalDescriptors)
  allocate_index_buffer(id, device)
  Vk.cmd_bind_index_buffer(cb, id.index_buffer[], 0, Vk.INDEX_TYPE_UINT32)
  write(descriptors.gset)
end
