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

struct Resource{T}
    type::ResourceType
    info::T
end

abstract type ResourceInfo end

struct BufferResourceInfo <: ResourceInfo
    size::Int
    usage::Vk.BufferUsageFlag
end

BufferResourceInfo(size) = BufferResourceInfo(size, Vk.BufferUsageFlag(0))

struct ImageResourceInfo <: ResourceInfo
    dims::Vector{Int}
    size_unit::SizeUnit
    format::Vk.Format
    usage::Vk.ImageUsageFlag
    aspect::Vk.ImageAspectFlag
end

function ImageResourceInfo(format; dims = [1., 1.], size_unit = SIZE_SWAPCHAIN_RELATIVE, usage = Vk.ImageUsageFlag(0), aspect = Vk.ImageAspectFlag(0))
    ImageResourceInfo(dims, size_unit, format, usage, aspect)
end

struct AttachmentResourceInfo <: ResourceInfo
    image_info::ImageResourceInfo
    AttachmentResourceInfo(image_info::ImageResourceInfo) = new(image_info)
end

AttachmentResourceInfo(args...; kwargs...) = AttachmentResourceInfo(ImageResourceInfo(args...; kwargs...))

struct Pass
    name::Symbol
    stages::Vk.PipelineStageFlag
end

Pass(stages) = Pass(gensym("Pass"), stages)

struct ResourceUsage
    type::ResourceType
    access::MemoryAccess
end

struct SynchronizationRequirements
    stages::Vk.PipelineStageFlag
    access::Vk.AccessFlag
    wait_semaphores::Vector{Vk.Semaphore}
end

SynchronizationRequirements(stages, access) = SynchronizationRequirements(stages, access, [])

"""
Frame graph implementation.

A frame graph has a list of virtual resources (buffers, images, attachments) that are
referenced by passes. They are turned into physical resources for the actual execution of those passes.

The frame graph uses two graph structures: a resource graph and an execution graph.

## Resource graph (bipartite, directed)

This bipartite graph has two types of vertices: passes and resources.
An edge from a resource to a pass describes a read dependency. An edge from a pass to a resource describes a write dependency.

Graph attributes:
- Resources:
    - `:name`: name of the resource
    - `:image_layout` (if the resource describes an image)
    - `:format` (if the resource describes either an image or an attachment)
    - `:aspect` (if the resource describes either an image or an attachment)
    - `:usage`
    - `:size` (if the resource describes a buffer)
    - `:vresource`: description as a virtual resource
    - `:presource`: physical resource
- Passes:
    - `:name`: name of the pass
- Edge between a resource and a pass (all directions)
    - `:image_layout` (if the resource describes an image)
    - `:usage`
    - `:aspect`
    - `:stage`: stages in which the resource is used

## Execution graph (directed, acyclic)

In this graph, vertices represent passes, and edges are resource dependencies between passes.
A topological sort of this graph represents a possible sequential execution order that respects execution dependencies.

Reusing the example above, the graph has three vertices: `gbuffer`, `lighting` and `adapt_luminance`.
`gbuffer` has five outgoing edges to `lighting`, each edge being labeled with a resource.
`lighting` has one outgoing edge to `adapt_luminance`.

This graph is generated just-in-time, to convert the resource graph into a linear sequence of passes.
"""
mutable struct FrameGraph
    device::Device
    resource_graph::MetaGraph{Int}
    resources::Dictionary{Symbol,Int}
    passes::Dictionary{Symbol,Int}
end

stages(fg::FrameGraph, i::Integer) = get_prop(fg.resource_graph, i, :stages)::Vk.PipelineStageFlag
attribute(rg::MetaDiGraph, i, key) = get_prop(rg, i, key)
pass_attribute(rg::MetaDiGraph, i::Integer, key::Symbol) = attribute(rg, i, key)
pass_attribute(fg::FrameGraph, name::Symbol, key::Symbol) = attribute(fg.resource_graph, fg.passes[name], key)
resource_attribute(rg::MetaDiGraph, i::Integer, key::Symbol) = attribute(rg, i, key)
resource_attribute(fg::FrameGraph, name::Symbol, key::Symbol) = attribute(fg.resource_graph, fg.resources[name], key)

virtual_image(g, idx) = resource_attribute(g, idx, :virtual_resource)::ImageResourceInfo
virtual_buffer(g, idx) = resource_attribute(g, idx, :virtual_resource)::BufferResourceInfo
virtual_attachment(g, idx) = resource_attribute(g, idx, :virtual_resource)::AttachmentResourceInfo

image(g, idx) = resource_attribute(g, idx, :physical_resource)::Image
buffer(g, idx) = resource_attribute(g, idx, :physical_resource)::Buffer
attachment(g, idx) = resource_attribute(g, idx, :physical_resource)::Attachment

image_layout(g, idx) = resource_attribute(g, idx, :image_layout)::Vk.ImageLayout
format(g, idx) = resource_attribute(g, idx, :format)::Vk.Format
aspect(g, idx) = resource_attribute(g, idx, :image_layout)::Vk.ImageAspectFlag
buffer_usage(g, idx) = resource_attribute(g, idx, :usage)::Vk.BufferUsageFlag
image_usage(g, idx) = resource_attribute(g, idx, :usage)::Vk.ImageUsageFlag
size(g, idx) = resource_attribute(g, idx, :size)::Int
name(g, idx) = attribute(g, idx, :name)::Symbol

function FrameGraph(device)
    FrameGraph(device, MetaGraph(), Dictionary{Symbol,Int}(), Dictionary{Symbol,Int}())
end

function add_pass!(fg::FrameGraph, name::Symbol, stages::Vk.PipelineStageFlag = Vk.PIPELINE_STAGE_ALL_GRAPHICS_BIT; clear_values = (0.1, 0.1, 0.1, 1.))
    !haskey(fg.passes, name) || error("Pass '$name' was already added to the frame graph. Passes can only be provided once.")
    g = fg.resource_graph
    add_vertex!(g)
    v = nv(g)
    set_prop!(g, v, :name, name)
    set_prop!(g, v, :stages, stages)
    set_prop!(g, v, :clear_values, clear_values)
    insert!(fg.passes, name, v)
    nothing
end

add_pass!(fg::FrameGraph, pass::Pass) = add_pass!(fg, pass.name, pass.stages)

function add_resource!(fg::FrameGraph, name::Symbol, resource::ResourceInfo, data = nothing)
    !haskey(fg.resources, name) || error("Resource '$name' was already added to the frame graph. Resources can only be provided once.")
    g = fg.resource_graph
    add_vertex!(g)
    v = nv(g)
    set_prop!(g, v, :name, name)
    insert!(fg.resources, name, v)
    if resource isa BufferResourceInfo
        set_prop!(g, v, :size, resource.size)
        set_prop!(g, v, :usage, resource.usage)
    else
        image_info = if resource isa AttachmentResourceInfo
            resource.image_info
        else
            resource
        end
        set_prop!(g, v, :dims, image_info.dims)
        set_prop!(g, v, :usage, image_info.usage)
        set_prop!(g, v, :aspect, image_info.aspect)
        set_prop!(g, v, :format, image_info.format)
    end
    if !isnothing(data)
        set_prop!(g, v, :imported, true)
        set_prop!(g, v, :data, data)
    end
    nothing
end

function add_resource_usage!(fg::FrameGraph, iter)
    for (pass, usage) in pairs(iter)
        add_resource_usage!(fg, pass, usage)
    end
end

function add_resource_usage!(fg::FrameGraph, pass::Symbol, usage::Dictionary{Symbol,ResourceUsage})
    haskey(fg.passes, pass) || error("Unknown pass '$(pass)'")
    v = fg.passes[pass]
    g = fg.resource_graph
    for (name, usage) in pairs(usage)
        haskey(fg.resources, name) || error("Unknown resource '$name'")
        i = fg.resources[name]
        add_edge!(g, i, v) || error("A resource usage between pass '$pass' and resource '$name' was already added to the frame graph. Resource usages can only be provided once.")
        set_prop!(g, i, v, :type, usage.type)
        set_prop!(g, i, v, :access, usage.access)
    end
end
add_resource_usage!(fg::FrameGraph, pass::Pass, usage::Dictionary{Symbol,ResourceUsage}) = add_resource_usage!(fg, pass.name, usage)

function resource_usages(ex::Expr)
    lines = @match ex begin
        Expr(:block, _...) => ex.args
        _ => [ex]
    end

    filter!(Base.Fix2(!isa, LineNumberNode), lines)

    usages = Dictionary{Symbol,Dictionary{Symbol,ResourceUsage}}()
    for line in lines
        (f, reads, writes) = @match line begin
            :($writes = $f($(reads...))) => @match writes begin
                Expr(:tuple, _...) => (f, reads, writes.args)
                ::Expr => (f, reads, [writes])
            end
            _ => error("Malformed expression, expected :(a, b = f(c, d)), got $line")
        end

        dict = Dictionary{Symbol,ResourceUsage}()
        for (name, type) in extract_resource_spec.(reads)
            insert!(dict, name, ResourceUsage(type, READ))
        end
        for (name, type) in extract_resource_spec.(writes)
            usage = if haskey(dict, name)
                ResourceUsage(type | dict[name].type, WRITE | READ)
            else
                ResourceUsage(type, WRITE)
            end
            set!(dict, name, usage)
        end
        insert!(usages, f, dict)
    end
    usages
end

macro resource_usages(ex)
    :($(resource_usages(ex)))
end

function extract_resource_spec(ex::Expr)
    @match ex begin
        :($r::Buffer::Vertex) => (r => RESOURCE_TYPE_VERTEX_BUFFER)
        :($r::Buffer::Index) => (r => RESOURCE_TYPE_INDEX_BUFFER)
        :($r::Buffer::Storage) => (r => RESOURCE_TYPE_BUFFER | RESOURCE_TYPE_STORAGE)
        :($r::Buffer::Uniform) => (r => RESOURCE_TYPE_BUFFER | RESOURCE_TYPE_UNIFORM)
        :($r::Color) => (r => RESOURCE_TYPE_COLOR_ATTACHMENT)
        :($r::Depth) => (r => RESOURCE_TYPE_DEPTH_ATTACHMENT)
        :($r::Stencil) => (r => RESOURCE_TYPE_STENCIL_ATTACHMENT)
        :($r::Depth::Stencil) || :($_::Stencil::Depth) => (r => RESOURCE_TYPE_DEPTH_ATTACHMENT | RESOURCE_TYPE_STENCIL_ATTACHMENT)
        :($r::Texture) => (r => RESOURCE_TYPE_TEXTURE)
        :($r::Image::Storage) => (r => RESOURCE_TYPE_IMAGE | RESOURCE_TYPE_STORAGE)
        :($r::Input) => (r => RESOURCE_TYPE_INPUT_ATTACHMENT)
        ::Symbol => error("Resource type annotation required: $ex")
        _ => error("Invalid or unsupported resource type annotation: $ex")
    end
end

function extract_resource_name(ex::Expr)
    @match ex begin
        :($r::$_::$_) => r
        :($r::$_) => r
        _ => error("Cannot extract resource name: $ex")
    end
end

function execution_graph(fg::FrameGraph)
    g = fg.resource_graph
    eg = SimpleDiGraph(length(fg.passes))
    for (i, pass) in enumerate(fg.passes)
        for resource in resources
            pass_indices = neighbors(g, resource)
            for j in pass_indices
                WRITE in get_prop(g, i, j, :access)::MemoryAccess || continue
                j ≠ i || error("Pass self-dependencies are not currently supported.")
                if j < i
                    # add a dependency from passes that write to this resource earlier
                    add_edge!(eg, j, i)
                end
            end
        end
    end
    eg
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
    indices = topological_sort_by_dfs(execution_graph(fg))
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

function aspect_bits(type::ResourceType)
    bits = Vk.ImageAspectFlag(0)
    RESOURCE_TYPE_COLOR_ATTACHMENT in type && (bits |= IMAGE_ASPECT_COLOR_BIT)
    RESOURCE_TYPE_DEPTH_ATTACHMENT in type && (bits |= IMAGE_ASPECT_DEPTH_BIT)
    RESOURCE_TYPE_STENCIL_ATTACHMENT in type && (bits |= IMAGE_ASPECT_STENCIL_BIT)
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

function resolve_attributes!(fg::FrameGraph)
    g = fg.resource_graph
    for edge in edges(g)
        type = get_prop(g, edge, :type)::ResourceType
        access = get_prop(g, edge, :access)::MemoryAccess
        stages = if has_prop(g, edge, :stages)
            get_prop(g, edge, :stages)::Vk.PipelineStageFlag
        else
            pass = edge.src in fg.passes ? edge.src : edge.dst
            get_prop(g, pass, :stages)::Vk.PipelineStageFlag
        end
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
