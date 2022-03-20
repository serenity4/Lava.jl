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

abstract type DrawCommand end

"""
Record that compacts action commands according to their state before flushing.

This allows to group e.g. draw calls that use the exact same rendering state.
"""
struct CompactRecord <: CommandRecord
  programs::Dictionary{Program,Dictionary{DrawState,Vector{Pair{DrawCommand,RenderTargets}}}}
  other_ops::Vector{LazyOperation}
  state::RefValue{DrawState}
  program::RefValue{Program}
  gd::GlobalData
  node::RenderNode
end

CompactRecord(gd::GlobalData, node::RenderNode) = CompactRecord(Dictionary(), [], Ref(DrawState()), Ref{Program}(), gd, node)

Base.show(io::IO, record::CompactRecord) = print(
  io,
  "CompactRecord(",
  length(record.programs),
  " programs, $(sum(x -> sum(length, values(x); init = 0), values(record.programs); init = 0)) draw commands)",
)

function set_program(record::CompactRecord, program::Program)
  record.program[] = program
end

function set_material(record::CompactRecord, @nospecialize(args...); alignment = 16)
  (; gd) = record

  # replace resource specifications with indices
  for (i, arg) in enumerate(args)
    if arg isa Texture
      @reset args[i] = texture_id!(record.fg, arg, record.pass)
    elseif arg isa Sampling
      @reset args[i] = sampler_id!(record.fg, arg, record.pass)
    end
  end

  sub = copyto!(gd.allocator, args, alignment)
  state = record.state[]
  record.state[] = @set state.push_data.material_data = device_address(sub)
end

function set_draw_state(record::CompactRecord, state::DrawState)
  record.state[] = state
end

draw_state(record::CompactRecord) = record.state[]

function draw(record::CompactRecord, targets::RenderTargets, vdata, idata; alignment = 16)
  (; gd) = record
  state = record.state[]

  # vertex data
  sub = copyto!(gd.allocator, align_blocks(vdata, alignment), alignment)
  record.state[] = @set state.push_data.vertex_data = device_address(sub)
  state = record.state[]

  # save draw command with its state
  program_draws = get!(Dictionary, record.programs, record.program[])
  commands = get!(Vector{DrawCommand}, program_draws, state)

  # index data
  first_index = length(gd.index_list) + 1
  append!(gd.index_list, idata)

  # draw call
  push!(commands, DrawIndexed(0, first_index:(first_index + length(idata) - 1), 1:1) => targets)
end

"""
Insert padding bytes after each element so that they
each start on an offset that is a multiple of `alignment`.
"""
function align_blocks(data::AbstractArray, alignment)
  size = sizeof(eltype(data))
  size % alignment == 0 && return reinterpret(UInt8, data)
  bytes = UInt8[]
  for el in data
    append!(bytes, reinterpret(UInt8, [el]))
    append!(bytes, zeros(UInt8, alignment - size % alignment))
  end
  bytes
end

struct Draw <: DrawCommand
  vertices::UnitRange{Int}
  instances::UnitRange{Int}
end

function apply(cb::CommandBuffer, draw::Draw)
  buffer = draw.parameters
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
  count::Int
end

function apply(cb::CommandBuffer, draw::DrawIndirect)
  buffer = draw.parameters
  Vk.cmd_draw_indirect(cb, buffer, offset(buffer), draw.count, stride(buffer))
end

struct DrawIndexed <: DrawCommand
  vertex_offset::Int
  indices::UnitRange{Int}
  instances::UnitRange{Int}
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
  count::Int
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
        info = pipeline_info(device, pass, program, state.render_state, state.program_state, baked.global_data.resources, targets)
        hash = request_pipeline(device, info)
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
  pass::RenderNode,
  program::Program,
  state::RenderState,
  invocation_state::ProgramInvocationState,
  resources::ResourceDescriptors,
  targets::RenderTargets,
)
  shader_stages = [Vk.PipelineShaderStageCreateInfo(shader) for shader in program.shaders]
  # Vertex data is retrieved from an address provided in the push constant.
  vertex_input_state = Vk.PipelineVertexInputStateCreateInfo([], [])
  rendering_state = Vk.PipelineRenderingCreateInfoKHR(0, format.(targets.color), format(targets.depth), format(targets.stencil))
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
  (; render_area::Vk.Rect2D) = pass
  (; x, y) = render_area.offset
  (; width, height) = render_area.extent
  viewport_state = Vk.PipelineViewportStateCreateInfo(viewports = [Vk.Viewport(x, height - y, width, -height, 0, 1)], scissors = [render_area])

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
    cull_mode = invocation_state.cull_mode,
  )
  nsamples = samples(first(targets.color))
  any(≠(nsamples) ∘ samples, targets.color) && error("Incoherent number of samples detected: $(samples.(targets.color))")
  multisample_state = Vk.PipelineMultisampleStateCreateInfo(Vk.SampleCountFlag(nsamples), false, 1.0, false, false)
  color_blend_state = Vk.PipelineColorBlendStateCreateInfo(false, Vk.LOGIC_OP_AND, attachments, ntuple(Returns(1.0f0), 4))
  layout = pipeline_layout(device, resources)
  Vk.GraphicsPipelineCreateInfo(
    shader_stages,
    rasterizer,
    layout,
    0,
    0;
    next = rendering_state,
    input_assembly_state,
    vertex_input_state,
    viewport_state,
    multisample_state,
    color_blend_state,
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
        reqs = BindRequirements(pipeline, state.push_data)
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
  populate_descriptor_sets!(gd)
end
