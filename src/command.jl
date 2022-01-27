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
  programs::Dictionary{Program,Dictionary{DrawState,Vector{Pair{DrawCommand,TargetAttachments}}}}
  other_ops::Vector{LazyOperation}
  state::Ref{DrawState}
  program::Ref{Program}
  fg::FrameGraph
  pass::Int
end

CompactRecord(fg::FrameGraph, pass::Int) = CompactRecord(Dictionary(), [], Ref(DrawState()), Ref{Program}(), fg, pass)

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
  (; gd) = record.fg.frame

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

function draw(record::CompactRecord, targets::TargetAttachments, vdata, idata; alignment = 16)
  (; gd) = record.fg.frame
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

function submit_pipelines!(device::Device, pass::RenderPass, record::CompactRecord)
  pipeline_hashes = Dictionary{ProgramInstance,UInt64}()
  for (program, calls) in pairs(record.programs)
    for (state, draws) in pairs(calls)
      for targets in unique!(last.(draws))
        rp = pass_attribute(record.fg.resource_graph, record.pass, :render_pass_handle)
        hash = submit_pipeline!(device, pass, program, state.render_state, state.program_state, record.fg.frame.gd.resources, targets, rp)
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
function submit_pipeline!(
  device::Device,
  pass::RenderPass,
  program::Program,
  state::RenderState,
  invocation_state::ProgramInvocationState,
  resources::ResourceDescriptors,
  targets::TargetAttachments,
  rp::Vk.RenderPass,
)
  shader_stages = Vk.PipelineShaderStageCreateInfo.(collect(program.shaders))
  # bindless: no vertex data
  vertex_input_state = Vk.PipelineVertexInputStateCreateInfo([], [])
  attachments = map(1:length(targets.color)) do attachment
    if isnothing(state.blending_mode)
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
  (; x, y) = pass.area.offset
  (; width, height) = pass.area.extent
  viewport_state = Vk.PipelineViewportStateCreateInfo(viewports = [Vk.Viewport(x, y, width, height, 0, 1)], scissors = [pass.area])
  rasterizer = Vk.PipelineRasterizationStateCreateInfo(
    false,
    false,
    invocation_state.polygon_mode,
    invocation_state.triangle_orientation,
    state.enable_depth_bias,
    1.0,
    0.0,
    0.0,
    1.0,
    cull_mode = invocation_state.face_culling,
  )
  multisample_state = Vk.PipelineMultisampleStateCreateInfo(Vk.SampleCountFlag(pass.samples), false, 1.0, false, false)
  color_blend_state = Vk.PipelineColorBlendStateCreateInfo(false, Vk.LOGIC_OP_AND, attachments, ntuple(Returns(1.0f0), 4))
  layout = pipeline_layout(device, resources)
  info = Vk.GraphicsPipelineCreateInfo(
    shader_stages,
    rasterizer,
    layout.handle,
    rp,
    0,
    0;
    vertex_input_state,
    multisample_state,
    color_blend_state,
    input_assembly_state,
    viewport_state,
  )
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

function initialize(cb::CommandBuffer, device::Device, gd::GlobalData, first_pipeline::Pipeline)
  allocate_index_buffer(gd, device)
  Vk.cmd_bind_index_buffer(cb, gd.index_buffer[], 0, Vk.INDEX_TYPE_UINT32)
  populate_descriptor_sets!(gd)
  Vk.cmd_bind_descriptor_sets(cb, Vk.PipelineBindPoint(first_pipeline.type), first_pipeline.layout, 0, [gd.resources.gset.set], [])
end
