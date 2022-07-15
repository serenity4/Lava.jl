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

abstract type RecordingMode end

struct DirectRecording <: RecordingMode end

struct StatefulRecording <: RecordingMode
  program::RefValue{Program}
  render_state::RefValue{RenderState}
  invocation_state::RefValue{ProgramInvocationState}
  data::RefValue{DrawData}
end

StatefulRecording() = StatefulRecording(Ref{Program}(), Ref(RenderState()), Ref(ProgramInvocationState()), Ref(DrawData()))

set_program(state::StatefulRecording, program::Program) = state.program[] = program
set_render_state(state::StatefulRecording, render_state::RenderState) = state.render_state[] = render_state
set_invocation_state(state::StatefulRecording, invocation_state::RenderState) = state.invocation_state[] = invocation_state
set_data(state::StatefulRecording, data::DrawData) = state.data[] = data
function set_material(state::StatefulRecording, material::UInt64)
  data = state.data[]
  set_data(state, @set data.material_data = material)
end
render_state(state::StatefulRecording) = state.render_state
invocation_state(state::StatefulRecording) = state.invocation_state

"""
Type that records command lazily, for them to be flushed into an Vulkan command buffer later.
"""
abstract type CommandRecord{R<:RecordingMode} <: LavaAbstraction end

abstract type DrawCommand end

"""
Record that compacts action commands according to their state before flushing.

This allows to group e.g. draw calls that use the exact same rendering state.
"""
struct CompactRecord{R<:RecordingMode} <: CommandRecord{R}
  node::RenderNode
  image_layouts::Dictionary{UUID,Vk.ImageLayout}
  gd::GlobalData
  resources::PhysicalResources
  programs::Dictionary{Program,Dictionary{DrawState,Vector{Pair{DrawCommand,RenderTargets}}}}
  other_ops::Vector{LazyOperation}
  layout::VulkanLayout
  mode::R
end

@forward CompactRecord{StatefulRecording}.mode (set_program, set_render_state, set_invocation_state, set_data, render_state, invocation_state)

function CompactRecord{R}(baked::BakedRenderGraph, node::RenderNode) where {R<:RecordingMode}
  image_layouts = Dictionary{ResourceUUID,Vk.ImageLayout}()
  for (uuid, usage) in pairs(baked.uses[node.uuid].images)
    insert!(image_layouts, uuid, image_layout(usage))
  end
  CompactRecord{R}(node, image_layouts, baked.global_data, baked.resources, baked.device.layout)
end

CompactRecord{R}(node::RenderNode, image_layouts, gd::GlobalData, resources::PhysicalResources, layout::VulkanLayout) where {R<:RecordingMode} =
  CompactRecord{R}(node, image_layouts, gd, resources, Dictionary(), [], Ref(DrawState()), Ref{Program}(), layout, R())

Base.show(io::IO, record::CompactRecord) = print(
  io,
  CompactRecord,
  '(',
  length(record.programs),
  " programs, $(sum(x -> sum(length, values(x); init = 0), values(record.programs); init = 0)) draw commands)",
)

function set_material(record::CompactRecord{StatefulRecording}, material::T) where {T}
  prog = record.mode.program[]
  # TODO: Look up what shaders use the material and create pointer resource accordingly, instead of using this weird heuristic.
  shader = vertex_shader(prog)
  !haskey(shader.source.typerefs, T) && (shader = fragment_shader(prog))
  device_ptr = allocate_pointer_resource!(record.gd.allocator, material, shader, record.layout)
  set_material(record.mode, device_ptr)
end

allocate_material(record::CompactRecord{DirectRecording}, program::Program, data) = allocate_material(record.gd.allocator, program, data, record.layout)
allocate_material(record::CompactRecord{StatefulRecording}, data) = allocate_material(record.gd.allocator, record.mode.program[], data, record.layout)

allocate_vertex_data(record::CompactRecord{DirectRecording}, program::Program, data) = allocate_vertex_data(record.gd.allocator, program, data, record.layout)
allocate_vertex_data(record::CompactRecord{StatefulRecording}, data) = allocate_vertex_data(record.gd.allocator, record.mode.program[], data, record.layout)

function draw(record::CompactRecord, command::DrawCommand, targets::RenderTargets, state::DrawState, program::Program)
  program_draws = get!(Dictionary, record.programs, program)
  commands = get!(Vector{DrawCommand}, program_draws, state)
  push!(commands, command => targets)
end

RenderTargets(rec::CompactRecord{DirectRecording}, color...; depth = nothing, stencil = nothing) = RenderTargets(rec, collect(color); depth, stencil)

function RenderTargets(rec::CompactRecord{DirectRecording}, color::AbstractVector; depth = nothing, stencil = nothing)
  color = map(color) do c
    isa(c, PhysicalAttachment) ? c : record.resources.attachments[uuid(c)]
  end
  isa(depth, LogicalAttachment) && (depth = record.resources.attachments[uuid(depth)])
  isa(stencil, LogicalAttachment) && (stencil = record.resources.attachments[uuid(stencil)])
  RenderTargets(color, depth, stencil)
end

function draw(record::CompactRecord{StatefulRecording}, vdata, idata, color...; depth = nothing, stencil = nothing, material = nothing, instances = 1:1)
  program = record.mode.program[]
  device_ptr = allocate_vertex_data(record, vdata, vertex_shader(program))
  data = record.mode.data[]
  set_data(record.mode, @set data.vertex_data = device_ptr)
  command = DrawIndexed(0, append!(record.gd, idata), instances)
  targets = RenderTargets(color...; depth, stencil)
  state = DrawState(render_state(record), invocation_state(record), record.mode.data[])
  draw(record, command, targets, state, program)
end

struct Draw <: DrawCommand
  vertices::UnitRange{Int64}
  instances::UnitRange{Int64}
end

function apply(cb::CommandBuffer, draw::Draw)
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
        info = pipeline_info(baked.device, record.node.render_area::Vk.Rect2D, program, state.render_state, state.program_state, baked.global_data.resources, targets)
        hash = request_pipeline(baked.device, info)
        set!(pipeline_hashes, ProgramInstance(program, state, targets), hash)
      end
    end
  end
  pipeline_hashes
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
  resources::ResourceDescriptors,
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

function Base.flush(cb::CommandBuffer, record::CompactRecord, device::Device, binding_state::BindState, pipeline_hashes)
  for op in record.other_ops
    apply(cb, op)
  end
  for (program, calls) in pairs(record.programs)
    for (state, draws) in pairs(calls)
      for (call, targets) in draws
        hash = pipeline_hashes[ProgramInstance(program, state, targets)]
        pipeline = device.pipeline_ht[hash]
        reqs = BindRequirements(pipeline, state.push_data, record.gd.resources.gset.set)
        bind(cb, reqs, binding_state)
        binding_state = reqs
        apply(cb, call)
      end
    end
  end
  binding_state
end

function initialize(cb::CommandBuffer, device::Device, gd::GlobalData)
  allocate_index_buffer(gd, device)
  Vk.cmd_bind_index_buffer(cb, gd.index_buffer[], 0, Vk.INDEX_TYPE_UINT32)
  write(gd.resources.gset)
end

function Texture(rec::CompactRecord, image::UUID, sampling = DEFAULT_SAMPLING)
  Texture(rec.resources.images[image], sampling)
end
Texture(rec::CompactRecord, image, sampling = DEFAULT_SAMPLING) = Texture(rec, uuid(image), sampling)
