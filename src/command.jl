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
    fg::FrameGraph
    pass::Int
end

CompactRecord(fg::FrameGraph, pass::Int) = CompactRecord(Dictionary(), [], Ref(DrawState()), Ref{Program}(), fg, pass)

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

function draw(record::CompactRecord, vdata, idata)
    (; gd) = record.fg.frame
    state = record.state[]

    program = deepcopy(get!(Dictionary{Program,Dictionary{DrawState,DrawCommand}}, record.programs, record.program[]))
    commands = get!(Vector{DrawCommand}, program, state)

    # vertex data
    sub = copyto!(gd.allocator, vdata)
    record.state[] = @set state.push_data.vertex_data = device_address(sub)

    # index data
    first_index = length(gd.index_list) + 1
    append!(gd.index_list, idata)

    # draw call
    push!(commands, DrawIndexed(0, first_index:first_index + length(idata), 1:1))
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

function Base.flush(cb::CommandBuffer, record::CompactRecord, device::Device, binding_state::BindState, pipeline_hashes)
    for op in record.other_ops
        apply(cb, op)
    end
    for (program, calls) in pairs(record.programs)
        for (state, call) in pairs(calls)
            hash = pipeline_hashes[(program, state)]
            pipeline = device.pipeline_ht[hash]
            reqs = BindRequirements(pipeline, state.push_data)
            bind(cb, reqs, binding_state)
            binding_state = reqs
            apply(cb, call)
        end
    end
    binding_state
end

function initialize(cb::Vk.CommandBuffer, gd::GlobalData, first_pipeline::Pipeline)
    gd.index_buffer[] = buffer(device(gd), gd.index_list)
    Vk.cmd_bind_index_buffer(cb, gd.ibuffer, 0, Vk.INDEX_TYPE_UINT32)
    populate_descriptor_sets!(gd)
    Vk.cmd_bind_descriptor_sets(cb, Vk.PipelineBindPoint(first_pipeline.type), first_pipeline.layout, 0, [gd.resources.set], [])
end
