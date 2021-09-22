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

function draw(record::CompactRecord, command::DrawCommand)
    program = get!(Dictionary{Program,Dictionary{DrawState,DrawCommand}}, record.programs, record.program)
    commands = get!(Vector{DrawCommand}, program, record.state)
    push!(commands, command)
end

struct DrawIndirect{B<:Buffer} <: DrawCommand
    parameters::B
    count::Int
end

function apply(cb::Vk.CommandBuffer, draw::DrawIndirect,)
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
Submit a pipeline create info for creation in the next batch.

A hash is returned to serve as the key to get the corresponding pipeline from the hash table.
"""
function submit_pipeline!(device::Device, pass::RenderPass, program::Program, state::RenderState, invocation_state::ProgramInvocationState)
    shader_stages = PipelineShaderStageCreateInfo.(program.shaders, program.specialization_constants)
    T = program.input_type
    vertex_input_state = PipelineVertexInputStateCreateInfo([VertexInputBindingDescription(T, 0)], vertex_input_attribute_descriptions(T, 0))
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
    pipeline_layout = Vk.PipelineLayout(program)
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
