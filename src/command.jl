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

"""
Apply all the commands recorded into `record` to the provided command buffer.
"""
function Base.flush(cb::Vk.CommandBuffer, record::CommandRecord) end

abstract type DrawCommand end

"""
Record that compacts action commands according to their state before flushing.

This allows to group e.g. draw calls that use the exact same rendering state.
"""
struct CompactRecord <: CommandRecord
    programs::Dictionary{Program, Dictionary{DrawState,Vector{DrawCommand}}}
    other_ops::Vector{LazyOperation}
    state::Ref{DrawState}
    program::Ref{Program}
end

CompactRecord() = CompactRecord(Dictionary(), [], Ref(DrawState()), Ref{Program}())

function set_program(record::CompactRecord, program::Program)
    record.program[] = program
end

function set_state(record::CompactRecord, state::DrawState)
    record.state[] = state
end

function set_state(record::CompactRecord, properties::NamedTuple)
    record.state[] = setproperties(record.state[], properties)
end

function draw(record::CompactRecord, command::DrawCommand)
    program = get!(Dictionary{Program,Dictionary{DrawState,DrawCommand}}, record.programs, record.program)
    commands = get!(Vector{DrawCommand}, program, record.state)
    push!(commands, command)
end

struct Draw <: DrawCommand
    vertices::UnitRange{Int}
    instances::UnitRange{Int}
end

function apply(cb::Vk.CommandBuffer, draw::Draw)
    buffer = draw.parameters
    Vk.cmd_draw(cb, draw.vertices.stop - draw.vertices.start, draw.instances.stop - draw.instances.start, draw.vertices.start - 1, draw.instances.start - 1)
end

struct DrawIndirect{B<:Buffer} <: DrawCommand
    parameters::B
    count::Int
end

function apply(cb::Vk.CommandBuffer, draw::DrawIndirect)
    buffer = draw.parameters
    Vk.cmd_draw_indirect(cb, buffer, offset(buffer), draw.count, stride(buffer))
end

struct DrawIndexed <: DrawCommand
    vertex_offset::Int
    indices::UnitRange{Int}
    instances::UnitRange{Int}
end

function apply(cb::Vk.CommandBuffer, draw::DrawIndexed)
    Vk.cmd_draw_indexed(cb, draw.indices.stop - draw.indices.start, draw.instances.stop - draw.instances.start, draw.indices.start - 1, draw.vertex_offset, draw.instances.start - 1)
end

struct DrawIndexedIndirect{B<:Buffer} <: DrawCommand
    parameters::B
    count::Int
end

function apply(cb::Vk.CommandBuffer, draw::DrawIndexedIndirect)
    buffer = draw.parameters
    Vk.cmd_draw_indexed_indirect(cb, buffer, offset(buffer), draw.count, stride(buffer))
end

struct FlushingState
    state::ProgramInterface
end

function submit_pipelines!(device::Device, pass::RenderPass, record::CompactRecord)
    pipeline_hashes = Dictionary{Tuple{Program,DrawState},UInt64}()
    for (program, calls) in pairs(record.programs)
        for state in keys(calls)
            hash = submit_pipeline!(device, pass, program, state.render_state, state.program_state)
            insert!(pipeline_hashes, (program, state), hash)
        end
    end
    pipeline_hashes
end

"""
Set of buffer handles for loading per-material and per-vertex data, along with global camera data.
"""
struct PushConstantData
    camera_data::UInt64
    material_data::UInt64
    vertex_data::UInt64
end

"""
Submit a pipeline create info for creation in the next batch.

A hash is returned to serve as the key to get the corresponding pipeline from the hash table.
"""
function submit_pipeline!(device::Device, pass::RenderPass, program::Program, state::RenderState, invocation_state::ProgramInvocationState, resources::ResourceDescriptors)
    shader_stages = PipelineShaderStageCreateInfo.(program.shader, program.shader.specialization_constants)
    # bindless: no vertex data
    vertex_input_state = PipelineVertexInputStateCreateInfo([], [])
    attachments = map(program.attachments) do attachment
        if isnothing(state.blending_mode)
            Vk.PipelineColorBlendAttachmentState(
                false,
                BLEND_FACTOR_SRC_ALPHA,
                BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                BLEND_OP_ADD,
                BLEND_FACTOR_SRC_ALPHA,
                BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                BLEND_OP_ADD;
                color_write_mask = state.color_write_mask,
            )
        else
            error("Color blending not supported")
        end
    end
    input_assembly_state = Vk.PipelineInputAssemblyStateCreateInfo(PrimitiveTopology(I), false)
    viewport_state = Vk.PipelineViewportStateCreateInfo(viewports = [Viewport(pass.area.offset..., pass.area.extent..., 0, 1)], scissors = [pass.area])
    rasterizer = Vk.PipelineRasterizationStateCreateInfo(false, false, invocation_state.polygon_mode, invocation_state.triangle_orientation, state.enable_depth_bias, 1.0, 0.0, 0.0, 1.0, cull_mode = invocation_state.face_culling)
    multisample_state = Vk.PipelineMultisampleStateCreateInfo(Vk.SampleCountFlag(pass.samples), false, 1.0, false, false)
    blend_state = Vk.PipelineColorBlendStateCreateInfo(false, Vk.LOGIC_OP_AND, attachments, ntuple(_ -> Vk.BLEND_FACTOR_ONE, 4))
    pipeline_layout = Vk.PipelineLayout(
        device,
        [resources.set.layout],
        [Vk.PushConstantRange(SHADER_STAGE_VERTEX, 0, sizeof(PushConstantData))],
    )
    info = Vk.GraphicsPipelineCreateInfo(
        shader_stages,
        rasterizer,
        pipeline_layout,
        render_pass,
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

function Base.flush(cb::Vk.CommandBuffer, record::CompactRecord, device::Device, binding_state::BindState, pipeline_hashes)
    for op in record.other_ops
        apply(cb, op)
    end
    for (program, calls) in pairs(record.programs)
        for (state, call) in pairs(calls)
            hash = pipeline_hashes[(program, state)]
            pipeline = device.pipeline_ht[hash]
            reqs = BindRequirements(program, state.program_state)
            bind(cb, reqs, binding_state)
            binding_state = reqs
            apply(cb, call)
        end
    end
end

struct GlobalData
    vbuffer::BufferBlock{MemoryBlock}
    ibuffer::BufferBlock{MemoryBlock}
    resources::ResourceDescriptors
end

function initialize_render(cb::Vk.CommandBuffer, gd::GlobalData, first_pipeline::Pipeline)
    Vk.cmd_bind_vertex_buffers(cb, [gd.vbuffer], [0])
    Vk.cmd_bind_index_buffer(cb, gd.ibuffer, 0, Vk.INDEX_TYPE_UINT32)
    Vk.cmd_bind_descriptor_sets(cb, Vk.PipelineBindPoint(first_pipeline.type), first_pipeline.layout, 0, [gd.resources.set], [])
end
