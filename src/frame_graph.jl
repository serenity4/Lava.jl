Vk.@bitmask_flag ResourceType::UInt32 begin
    RESOURCE_TYPE_VERTEX_BUFFER = 1
    RESOURCE_TYPE_INDEX_BUFFER = 2
    RESOURCE_TYPE_COLOR_ATTACHMENT = 4
    RESOURCE_TYPE_DEPTH_ATTACHMENT = 8
    RESOURCE_TYPE_STENCIL_ATTACHMENT = 16
    RESOURCE_TYPE_INPUT_ATTACHMENT = 32
    RESOURCE_TYPE_TEXTURE = 64
    RESOURCE_TYPE_BUFFER = 128
    RESOURCE_TYPE_IMAGE = 256
    RESOURCE_TYPE_DYNAMIC = 512
    RESOURCE_TYPE_STORAGE = 1024
    RESOURCE_TYPE_TEXEL = 2048
    RESOURCE_TYPE_UNIFORM = 4096
    RESOURCE_TYPE_SAMPLER = 8192
end

struct Resource
    type::ResourceType
    descriptor_type::Optional{Vk.DescriptorType}
end

abstract type ResourceInfo end

struct BufferResourceInfo <: ResourceInfo
    size::Int
    usage::Vk.BufferUsageFlag
end

struct ImageResourceInfo <: ResourceInfo
    dims::Vector{Int}
    size_unit::SizeUnit
    format::Vk.Format
    usage::Vk.ImageUsageFlag
    aspect::Vk.ImageAspectFlag
end

struct AttachmentResourceInfo <: ResourceInfo
    image_info::ImageResourceInfo
end

AttachmentResourceInfo(args...) = AttachmentResourceInfo(ImageResourceInfo(args...))

struct Pass
    writes::Vector{Int}
    reads::Vector{Int}
end

struct SynchronizationRequirements
    stages::Vk.PipelineStageFlag
    access::Vk.AccessFlag
    wait_semaphores::Vector{Vk.Semaphore}
end

"""
Frame graph implementation.

A frame graph has a list of virtual resources (buffers, images, attachments) that are
referenced by passes. They are turned into physical resources for the actual execution of those passes.

The frame graph possesses two graph structure: a resource graph and an execution graph.

## Dependency graph (directed, acyclic)

In this graph, vertices represent passes, and edges are resource dependencies between passes.
A topological sort of this graph represents a possible execution order that respects execution dependencies.

Reusing the example above, the graph has three vertices: `gbuffer`, `lighting` and `adapt_luminance`.
`gbuffer` has five outgoing edges to `lighting`, each edge being labeled with a resource.
`lighting` has one outgoing edge to `adapt_luminance`.

## Resource graph (bipartite, directed)

This bipartite graph has two types of vertices: passes and resources.
An edge from a resource to a pass describes a read dependency. An edge from a pass to a resource describes a write dependency.
"""
struct FrameGraph
    device::Device
    resource_graph::SimpleDiGraph{Int}
    execution_graph::SimpleDiGraph{Int}
    passes::Vector{Pass}

    # virtual resources
    buffers::Vector{BufferResourceInfo}
    images::Vector{ImageResourceInfo}
    attachments::Vector{AttachmentResourceInfo}

    # physical resources
    physical_images::Vector{Image}
    physical_buffers::Vector{Buffer}
    physical_attachments::Vector{Attachment}
end

is_resource(fg::FrameGraph, i) = i > length(fg.passes)
is_pass(fg::FrameGraph, i) = !is_resource(fg, i)

is_buffer(fg::FrameGraph, i) = i ≤ length(fg.buffers)
is_image(fg::FrameGraph, i) = length(fg.buffers) < i ≤ length(fg.buffers) + length(fg.images)
is_attachment(fg::FrameGraph, i) = i > length(fg.buffers) + length(fg.images)

get_image(fg::FrameGraph, i) = fg.images[fg] + i

function FrameGraph(device::Device, passes, buffers, images, attachments)
    np = length(passes)
    nr = length(attachments) + length(images) + length(buffers)

    rg = SimpleDiGraph(np + nr)
    for (i, pass) in enumerate(passes)
        for read in pass.reads
            add_edge!(g, np + read, i)
        end
        for write in pass.writes
            add_edge!(g, i, np + write)
        end
    end

    eg = SimpleDiGraph(np)
    for (i, pass) in enumerate(passes)
        for resource in [pass.reads; pass.writes]
            pass_indices = inneighbors(resource + np)
            for j in pass_indices
                j ≠ i || error("Pass self-dependencies are not currently supported.")
                if j < i
                    # add a dependency from passes that write to this resource earlier
                    add_edge!(eg, j, i)
                end
            end
        end
    end

    FrameGraph(device, rg, eg, passes, buffers, images, attachments, [], [], [])
end

function synchronize_before(cb::Vk.CommandBuffer, fg::FrameGraph, pass::Pass)
    image_memory_barriers = Vk.ImageMemoryBarriers[]
    buffer_memory_barriers = Vk.BufferMemoryBarriers[]
    memory_barriers = Vk.MemoryBarriers[]
    for read in pass.reads
        if is_image(fg, read)
            image = fg.images[read]
        end
    end
end

"""
Submit rendering commands to a device.

A command buffer is recorded, which may be split into multiple ones to take advantage of multithreading,
and is then submitted them to the provided device.
A fence and/or a semaphore can be provided to synchronize with the application or other commands.
"""
function render(device, fg::FrameGraph; fence = nothing, semaphore = nothing)
    cb = get_command_buffer(device, fg)
    @record cb begin
        for pass in sort_passes(fg)
            synchronize_before(cb, fg, pass)
            begin_render_pass(cb, pass)
            record(cb, pass)
            Vk.cmd_end_render_pass()
            synchronize_after(cb, fg, pass)
        end
    end
    submit_info = Vk.SubmitInfo([], [cb], isnothing(semaphore) ? [semaphore] : [])
    Vk.queue_submit(device, submit_info)
end

function sort_passes(fg::FrameGraph)
    indices = topological_sort_by_dfs(fg.dependency_graph)
    fg.passes[indices]
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

"""
Deduce the Vulkan usage, layout and access flags form a resource given its type, stage and access.

The idea is to reconstruct information like `Vk.ACCESS_COLOR_ATTACHMENT_READ_BIT` and `Vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` from a more decoupled description.
"""
function image_layout(type::ResourceType, stage::Vk.PipelineStageFlag, access::MemoryAccess)
    @match (type, access) begin
        (&RESOURCE_TYPE_COLOR_ATTACHMENT, &READ) => Vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        (&RESOURCE_TYPE_COLOR_ATTACHMENT, &WRITE) => Vk.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        (&RESOURCE_TYPE_COLOR_ATTACHMENT || &RESOURCE_TYPE_IMAGE || RESOURCE_TYPE_TEXTURE, &(READ | WRITE)) => Vk.IMAGE_LAYOUT_GENERAL
        (&RESOURCE_TYPE_DEPTH_ATTACHMENT, &READ) => Vk.IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL
        (&RESOURCE_TYPE_DEPTH_ATTACHMENT, &WRITE) => Vk.IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
        (&RESOURCE_TYPE_STENCIL_ATTACHMENT, &READ) => Vk.IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL
        (&RESOURCE_TYPE_STENCIL_ATTACHMENT, &WRITE) => Vk.IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL
        (&(RESOURCE_TYPE_DEPTH_ATTACHMENT | RESOURCE_TYPE_STENCIL_ATTACHMENT), &READ) => Vk.IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
        (&(RESOURCE_TYPE_DEPTH_ATTACHMENT | RESOURCE_TYPE_STENCIL_ATTACHMENT), &WRITE) => Vk.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        (&RESOURCE_TYPE_INPUT_ATTACHMENT || &RESOURCE_TYPE_TEXTURE || &RESOURCE_TYPE_IMAGE, &READ) => Vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        _ => error("Unsupported combination of type $type and access $access")
    end
end

function usage_bits(type::ResourceType, access::MemoryAccess)
    @match type begin
        &RESOURCE_TYPE_COLOR_ATTACHMENT => Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT
        &RESOURCE_TYPE_DEPTH_ATTACHMENT || &RESOURCE_TYPE_STENCIL_ATTACHMENT => Vk.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
        &RESOURCE_TYPE_INPUT_ATTACHMENT => Vk.IMAGE_USAGE_INPUT_ATTACHMENT_BIT
        &RESOURCE_TYPE_VERTEX_BUFFER => Vk.BUFFER_USAGE_VERTEX_BUFFER_BIT
        &RESOURCE_TYPE_INDEX_BUFFER => Vk.BUFFER_USAGE_INDEX_BUFFER_BIT
        _ => @match (type, access) begin
            (&RESOURCE_TYPE_BUFFER, &READ) => Vk.BUFFER_USAGE_UNIFORM_BUFFER_BIT
            (&RESOURCE_TYPE_BUFFER, &WRITE) => Vk.BUFFER_USAGE_STORAGE_BUFFER_BIT
            (&RESOURCE_TYPE_BUFFER, &(READ | WRITE)) => Vk.BUFFER_USAGE_UNIFORM_BUFFER_BIT | Vk.BUFFER_USAGE_STORAGE_BUFFER_BIT
            (&RESOURCE_TYPE_IMAGE, &WRITE) => Vk.IMAGE_USAGE_STORAGE_BIT
            _ => error("Unsupported combination of type $type and access $access")
        end
    end
end

const SHADER_STAGES = |(
    Vk.PIPELINE_STAGE_VERTEX_SHADER_BIT,
    Vk.PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT,
    Vk.PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT,
    Vk.PIPELINE_STAGE_GEOMETRY_SHADER_BIT,
    Vk.PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
    Vk.PIPELINE_STAGE_COMPUTE_SHADER_BIT,
)

function access_bits(type::ResourceType, access::MemoryAccess, stage::Vk.PipelineStageFlag)
    bits = Vk.AccessFlag(0)
    RESOURCE_TYPE_VERTEX_BUFFER in type && (bits |= Vk.ACCESS_VERTEX_ATTRIBUTE_READ_BIT)
    RESOURCE_TYPE_INDEX_BUFFER in type && (bits |= Vk.ACCESS_INDEX_READ_BIT)
    if RESOURCE_TYPE_COLOR_ATTACHMENT in type
        READ in access && (bits |= Vk.ACCESS_COLOR_ATTACHMENT_READ_BIT)
        WRITE in access && (bits |= Vk.ACCESS_COLOR_ATTACHMENT_WRITE_BIT)
    end
    if (RESOURCE_TYPE_DEPTH_ATTACHMENT in type || RESOURCE_TYPE_STENCIL_ATTACHMENT in type)
        #TODO: support mixed access modes (depth write, stencil read and vice-versa)
        READ in access && (bits |= Vk.ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT)
        WRITE in access && (bits |= Vk.ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
    end
    RESOURCE_TYPE_INPUT_ATTACHMENT in type && (bits |= Vk.ACCESS_INPUT_ATTACHMENT_READ_BIT)
    if RESOURCE_TYPE_BUFFER in type && !iszero(stage & SHADER_STAGES)
        access == READ && (bits |= Vk.ACCESS_UNIFORM_READ_BIT)
        WRITE in access && (bits |= Vk.ACCESS_SHADER_WRITE_BIT)
    end
    bits
end

function descriptor_type(type::ResourceType, access::MemoryAccess)
    @match access, type begin
        &(RESOURCE_TYPE_IMAGE | RESOURCE_TYPE_STORAGE) => Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE
        &(RESOURCE_TYPE_TEXEL | RESOURCE_TYPE_BUFFER) => Vk.DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER
        &(RESOURCE_TYPE_UNIFORM | RESOURCE_TYPE_BUFFER) => Vk.DESCRIPTOR_TYPE_UNIFORM_BUFFER
        &(RESOURCE_TYPE_UNIFORM | RESOURCE_TYPE_BUFFER | RESOURCE_TYPE_DYNAMIC) => Vk.DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
        &(RESOURCE_TYPE_UNIFORM | RESOURCE_TYPE_TEXEL | RESOURCE_TYPE_BUFFER) => Vk.DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
        &(RESOURCE_TYPE_STORAGE | RESOURCE_TYPE_BUFFER) => Vk.DESCRIPTOR_TYPE_STORAGE_BUFFER
        &(RESOURCE_TYPE_STORAGE | RESOURCE_TYPE_BUFFER | RESOURCE_TYPE_DYNAMIC) => Vk.DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC
        &RESOURCE_TYPE_TEXTURE => Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE
        &(RESOURCE_TYPE_TEXTURE | ESOURCE_TYPE_SAMPLER) => Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
        &RESOURCE_TYPE_SAMPLER => Vk.DESCRIPTOR_TYPE_SAMPLER
        _ => error("Unsupported combination of type $type and access $access")
    end
end

function create_physical_resources!(fg::FrameGraph)
    for buffer in fg.buffers
        push!(fg.physical_buffers, BufferBlock(fg.device, buffer.size, buffer.usage))
    end
    for image in fg.images
        push!(fg.physical_images, ImageBlock(fg.device, image.dims, image.format, image.usage))
    end
    for attachment in fg.attachments
        push!(fg.physical_attachments, Attachment(ImageView(physical_image(fg, attachment))))
    end
end
