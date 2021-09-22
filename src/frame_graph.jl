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

@enum ResourceClass::Int8 begin
    RESOURCE_CLASS_BUFFER = 1
    RESOURCE_CLASS_IMAGE = 2
    RESOURCE_CLASS_ATTACHMENT = 3
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
    - `:format` (if the resource describes either an image or an attachment)
    - `:usage`
    - `:size` (if the resource describes a buffer)
    - `:vresource`: description as a virtual resource
    - `:presource`: physical resource
    - `:current_layout` (if the resource describes an image)
    - `:last_write`: access bits and pipeline stages associated to the latest pass that wrote to the resource
    - `:synchronization_state`: access bits and pipeline stages of all passes that required to synchronize with this resource
        since the last write.
- Passes:
    - `:name`: name of the pass
- Edge between a resource and a pass (all directions)
    - `:image_layout` (if the resource describes an image)
    - `:usage`
    - `:aspect`
    - `:stage`: stages in which the resource is used
    - `:clear_value`: clear value, if applicable. If this property is absent and a load operation takes place on this resource in this pass,
        then the contents are preserved.

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

attribute(rg::AbstractMetaGraph, i, key) = get_prop(rg, i, key)
pass_attribute(rg::AbstractMetaGraph, i::Integer, key::Symbol) = attribute(rg, i, key)
pass_attribute(fg::FrameGraph, name::Symbol, key::Symbol) = attribute(fg.resource_graph, fg.passes[name], key)
resource_attribute(rg::AbstractMetaGraph, i::Integer, key::Symbol) = attribute(rg, i, key)
resource_attribute(fg::FrameGraph, name::Symbol, key::Symbol) = attribute(fg.resource_graph, fg.resources[name], key)

virtual_image(g, idx) = resource_attribute(g, idx, :virtual_resource)::ImageResourceInfo
virtual_buffer(g, idx) = resource_attribute(g, idx, :virtual_resource)::BufferResourceInfo
virtual_attachment(g, idx) = resource_attribute(g, idx, :virtual_resource)::AttachmentResourceInfo

image(g, idx) = resource_attribute(g, idx, :physical_resource)::Image
buffer(g, idx) = resource_attribute(g, idx, :physical_resource)::Buffer
attachment(g, idx) = resource_attribute(g, idx, :physical_resource)::Attachment

type(g, i, j) = attribute(g, Edge(i, j), :type)::ResourceType
access(g, i, j) = attribute(g, Edge(i, j), :access)::MemoryAccess
access_bits(g, i, j) = attribute(g, Edge(i, j), :access_bits)::Vk.AccessFlag
image_layout(g, i, j) = attribute(g, Edge(i, j), :image_layout)::Vk.ImageLayout
clear_value(g, i, j) = attribute(g, Edge(i, j), :clear_value)::Vk.ClearValue
aspect(g, i, j) = attribute(g, Edge(i, j), :aspect)::Vk.ImageAspectFlag
stages(g, i, j) = attribute(g, Edge(i, j), :stages)::Vk.PipelineStageFlag

current_layout(g, idx) = resource_attribute(g, idx, :current_layout)::Vk.ImageLayout
format(g, idx) = resource_attribute(g, idx, :format)::Vk.Format
buffer_usage(g, idx) = resource_attribute(g, idx, :usage)::Vk.BufferUsageFlag
image_usage(g, idx) = resource_attribute(g, idx, :usage)::Vk.ImageUsageFlag
size(g, idx) = resource_attribute(g, idx, :size)::Int
name(g, idx) = attribute(g, idx, :name)::Symbol
last_write(g, idx) = resource_attribute(g, idx, :last_write)::Pair{Vk.AccessFlag, Vk.PipelineStageFlag}
synchronization_state(g, idx) = resource_attribute(g, idx, :synchronization_state)::Dictionary{Vk.AccessFlag,Vk.PipelineStageFlag}
stages(g, idx) = pass_attribute(g, idx, :stages)::Vk.PipelineStageFlag

function FrameGraph(device)
    FrameGraph(device, MetaGraph(), Dictionary(), Dictionary())
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
    for pass in fg.passes
        pass_indices = neighbors(g, pass)
        for j in pass_indices
            WRITE in get_prop(g, i, j, :access)::MemoryAccess || continue
            j ≠ i || error("Pass self-dependencies are not currently supported.")
            if j < i
                # add a dependency from passes that write to this resource earlier
                add_edge!(eg, j, i)
            end
        end
    end
    eg
end

"""
Submit rendering commands to a device.

A command buffer is recorded, which may be split into multiple ones to take advantage of multithreading,
and is then submitted them to the provided device.
A fence and/or a semaphore can be provided to synchronize with the application or other commands.
"""
function render(device, fg::FrameGraph; fence = nothing, semaphore = nothing)
    records = CompactRecord[]
    pipeline_hashes = Dictionary{Tuple{Program,DrawState},UInt64}()
    passes = sort_passes(fg)

    # record commands and submit pipelines for creation
    for pass in passes
        record = CompactRecord(device)
        record_render_pass(record, fg, pass)
        push!(records, record)
        merge!(pipeline_hashes, submit_pipelines!(device, fg, pass))
    end

    batch_create!(device.pipeline_ht, device.pending_pipelines) do infos
        handles = unwrap(Vk.create_graphics_pipelines(device, infos))
        map(zip(handles, infos)) do (handle, info)
            Pipeline(handle, Vk.PIPELINE_BIND_POINT_GRAPHICS, info.pipeline_layout)
        end
    end

    # fill command buffer with synchronization commands & recorded commands
    cb = get_command_buffer(device)
    @record cb begin
        for (pass, record) in zip(passes, records)
            begin_render_pass(cb, fg, passes, pass)
            synchronize_before(cb, fg, pass)
            flush(cb, record, device, BindState(), pipeline_hashes)
            Vk.cmd_end_render_pass()
            synchronize_after(cb, fg, pass)
        end
    end

    # submit rendering work
    submit_info = Vk.SubmitInfo2KHR([], [Vk.CommandBufferSubmitInfoKHR(cb, 0)], isnothing(semaphore) ? [Vk.SemaphoreSignalInfo(semaphore, 0)] : [])
    unwrap(Vk.queue_submit_2_khr(device, submit_info))
    nothing
end

function sort_passes(fg::FrameGraph)
    indices = topological_sort_by_dfs(execution_graph(fg))
    collect(values(fg.passes))[indices]
end

"""
Build barriers for all resources that require it.

Requires the extension `VK_KHR_synchronization2`.
"""
function synchronize_before(cb::Vk.CommandBuffer, fg::FrameGraph, pass::Integer)
    g = fg.resource_graph
    deps = Vk.DependencyInfoKHR([], [], [])
    for resource in neighbors(fg, pass)
        (req_access, req_stages) = access(g, pass, resource) => stages(g, pass)
        # if the resource was not written to recently, no synchronization is required
        if has_prop(g, resource, :last_write)
            (r_access, p_stages) = last_write(g, resource)
            resource_type = type(g, pass, resource)
            req_access_bits = access_bits(resource_type, req_access, req_stages)
            sync_state = synchronization_state(g, resource)
            synced_stages = PipelineStageFlag(0)
            barrier_needed = true
            for (access, stages) in pairs(sync_state)
                if covers(access, req_access_bits)
                    synced_stages |= (stages & req_stages)
                end
                if synced_stages == req_stages
                    barrier_needed = false
                    break
                else
                    # keep only stages that haven't been synced
                    req_stages &= ~synced_stages
                end
            end
            if barrier_needed
                @switch class = ResourceClass(resource_type) begin
                    @case &RESOURCE_CLASS_BUFFER
                        buff = buffer(g, resource)
                        barrier = Vk.BufferMemoryBarrier2KHR(p_stages, req_stages, r_access, req_access_bits, 0, 0, buff, offset(buff), size(buff))
                        push!(deps.buffer_memory_barriers, barrier)
                    @case &RESOURCE_CLASS_IMAGE || &RESOURCE_CLASS_ATTACHMENT
                        view = if class == RESOURCE_CLASS_IMAGE
                            ImageView(image(g, resource))
                        else
                            attachment(g, resource).view
                        end
                        new_layout = image_layout(g, pass, resource)
                        range = subresource_range(view.aspect, view.mip_range, view.layer_range)
                        barrier = Vk.ImageMemoryBarrier(r_access, req_access_bits, current_layout(g, resource), new_layout, 0, 0, range)
                        push!(deps.image_memory_barriers, barrier)
                        set_prop!(g, resource, :current_layout, new_layout)
                end
                set!(sync_state, req_access_bits, req_stages)
            end
        end
        if WRITE in req_access
            set_prop!(g, resource, :last_write, req_access_bits => req_stages)
        end
    end
    Vk.cmd_pipeline_barrier_2_khr(cb, deps)
end

function begin_render_pass(cb, fg::FrameGraph, pass::Integer)
    g = fg.resource_graph
    attachment_nodes = (resource for resource in neighbors(g, pass) if ResourceClass(type(g, pass, resource)) == RESOURCE_CLASS_ATTACHMENT)
    attachms = Attachment[]
    attach_descs = Vk.AttachmentDescription2[]
    clear_values = Vk.ClearValue[]

    subpass_descriptions = Vk.SubpassDescription2[]
    subpass_dependencies = Vk.SubpassDependency2[]

    for (i, node) in enumerate(attachment_nodes)
        # render pass (global) attachment info
        attachm = attachment(g, pass, i)
        push!(attachms, attach)
        clear = has_prop(g, pass, node, :clear_value)
        push!(attach_descs, Vk.AttachmentDescription2(attachm, clear, image_layout(g, pass, node), image_layout(g, pass, node)))
        clear_val = if clear
            clear_value(g, pass, node)
        else
            Vk.ClearValue(Vk.ClearColorValue(ntuple(Returns(0f0), 4)))
        end
        push!(clear_values, clear_value)

        # subpass (local) attachment info
        # only one subpass per pass is currently supported
        # we could later look up an internal subpass graph to support more
        for subpass in 1:1
            stage_flags = stages(g, pass)
            pipeline_bind_point = Vk.PIPELINE_STAGE_COMPUTE_SHADER_BIT in stage_flags ? Vk.PIPELINE_BIND_POINT_COMPUTE : Vk.PIPELINE_BIND_POINT_GRAPHICS
            color_attachments = Vk.AttachmentReference2[]
            depth_stencil_attachment = C_NULL
            input_attachments = Vk.AttachmentReference2[]
            resource_type = type(g, pass, resource)
            ref = Vk.AttachmentReference2(i, image_layout(f, pass, node), aspect(g, i, j))
            if RESOURCE_TYPE_COLOR_ATTACHMENT in resource_type
                push!(color_attachments, ref)
            end
            if RESOURCE_TYPE_DEPTH_ATTACHMENT in resource_type || RESOURCE_TYPE_STENCIL_ATTACHMENT in resource_type
                depth_stencil_attachment = ref
            end
            if RESOURCE_TYPE_INPUT_ATTACHMENT in resource_type
                push!(input_attachments, ref)
            end
            push!(subpass_descriptions, Vk.SubpassDescription2(pipeline_bind_point, 0, input_attachments, color_attachments, []; depth_stencil_attachment))

            acc_bits = access_bits(g, i, j)
            push!(subpass_dependencies, Vk.SubpassDependency2(Vk.SUBPASS_EXTERNAL, subpass - 1, 0; dst_stage_mask = stage_flags, dst_access_mask = acc_bits))
        end
    end
    render_pass = RenderPass(fg, pass)
    render_pass_handle = Vk.RenderPass(fg.device, attachment_descriptions, subpass_descriptions, subpass_dependencies)
    (; width, height) = render_pass.area.extent
    fb = Vk.Framebuffer(fg.device, render_pass_handle, attachms, width, height, 1)
    Vk.cmd_begin_render_pass(cb, Vk.RenderPassBeginInfo(render_pass_handle, fb, render_pass.area, clear_values))
end

function synchronize_after(cb, fg, pass)
    nothing
end

"""
Deduce the Vulkan usage, layout and access flags form a resource given its type, stage and access.

The idea is to reconstruct information like `Vk.ACCESS_COLOR_ATTACHMENT_READ_BIT` and `Vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` from a more decoupled description.
"""
function image_layout(type::ResourceType, access::MemoryAccess, stage::Vk.PipelineStageFlag)
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

function buffer_usage_bits(type::ResourceType, access::MemoryAccess)
    bits = Vk.BufferUsageFlag(0)

    RESOURCE_TYPE_BUFFER | RESOURCE_TYPE_STORAGE in type && (bits |= Vk.BUFFER_USAGE_STORAGE_BUFFER_BIT)
    RESOURCE_TYPE_VERTEX_BUFFER in type && (bits |= Vk.BUFFER_USAGE_VERTEX_BUFFER_BIT)
    RESOURCE_TYPE_INDEX_BUFFER in type && (bits |= Vk.BUFFER_USAGE_INDEX_BUFFER_BIT)
    RESOURCE_TYPE_BUFFER in type && access == READ && (bits |= Vk.BUFFER_USAGE_UNIFORM_BUFFER_BIT)

    bits
end

function image_usage_bits(type::ResourceType, access::MemoryAccess)
    bits = Vk.ImageUsageFlag(0)

    RESOURCE_TYPE_COLOR_ATTACHMENT in type && (bits |= Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
    (RESOURCE_TYPE_DEPTH_ATTACHMENT in type || RESOURCE_TYPE_STENCIL_ATTACHMENT in type) && (bits |= Vk.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
    RESOURCE_TYPE_INPUT_ATTACHMENT in type && (bits |= Vk.IMAGE_USAGE_INPUT_ATTACHMENT_BIT)
    RESOURCE_TYPE_TEXTURE in type && (bits |= Vk.IMAGE_USAGE_SAMPLED_BIT)
    RESOURCE_TYPE_IMAGE in type && WRITE in access && (bits |= Vk.IMAGE_USAGE_STORAGE_BIT)

    bits
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
    RESOURCE_TYPE_TEXTURE in type && READ in access && (bits |= Vk.ACCESS_SHADER_READ_BIT)
    RESOURCE_TYPE_TEXTURE in type && WRITE in access && (bits |= Vk.ACCESS_SHADER_WRITE_BIT)
    bits
end

function aspect_bits(type::ResourceType)
    bits = Vk.ImageAspectFlag(0)
    RESOURCE_TYPE_COLOR_ATTACHMENT in type && (bits |= Vk.IMAGE_ASPECT_COLOR_BIT)
    RESOURCE_TYPE_DEPTH_ATTACHMENT in type && (bits |= Vk.IMAGE_ASPECT_DEPTH_BIT)
    RESOURCE_TYPE_STENCIL_ATTACHMENT in type && (bits |= Vk.IMAGE_ASPECT_STENCIL_BIT)
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
    for pass in fg.passes
        for resource in neighbors(g, pass)
            resource_type = type(g, pass, resource)
            resource_access = access(g, pass, resource)
            pipeline_stages = if has_prop(g, pass, resource, :stages)
                stages(g, pass, resource)
            else
                stages(g, pass)
            end
            # class-independent attributes
            acc_bits = access_bits(resource_type, resource_access, pipeline_stages)
            set_prop!(g, pass, resource, :access_bits, acc_bits)

            # class-dependent attributes
            resource_class = ResourceClass(resource_type)
            if resource_class == RESOURCE_CLASS_IMAGE || resource_class == RESOURCE_CLASS_ATTACHMENT
                usage = image_usage_bits(resource_type, resource_access)
                set_prop!(g, pass, resource, :usage, usage)
                layout = image_layout(resource_type, resource_access, pipeline_stages)
                set_prop!(g, pass, resource, :image_layout, layout)
                aspect = aspect_bits(resource_type)
                set_prop!(g, pass, resource, :aspect, aspect)
                set_prop!(g, resource, :usage, usage | image_usage(g, resource))
            else
                usage = buffer_usage_bits(resource_type, resource_access)
                set_prop!(g, pass, resource, :usage, usage)
                set_prop!(g, resource, :usage, usage | buffer_usage(g, resource))
            end
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

"""
Return whether `x` covers accesses of type `y`; that is, whether guarantees about memory access operations in `x` induce guarantees about memory access operations in `y`.

```jldoctest
julia> covers(Vk.ACCESS_MEMORY_WRITE_BIT, Vk.ACCESS_SHADER_WRITE_BIT)
true

julia> covers(Vk.ACCESS_SHADER_READ_BIT, Vk.ACCESS_UNIFORM_READ_BIT)
true

julia> covers(Vk.ACCESS_SHADER_WRITE_BIT, Vk.ACCESS_MEMORY_WRITE_BIT)
false
```

"""
function covers(x::Vk.AccessFlag, y::Vk.AccessFlag)
    y in x && return true
    if Vk.ACCESS_MEMORY_READ_BIT in x
        if Vk.ACCESS_MEMORY_READ_BIT ∉ y
            x &= ~Vk.ACCESS_MEMORY_READ_BIT
        end
        x |= |(
            Vk.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
            Vk.ACCESS_COLOR_ATTACHMENT_READ_BIT,
            Vk.ACCESS_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT,
            Vk.ACCESS_COMMAND_PREPROCESS_READ_BIT_NV,
            Vk.ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT,
            Vk.ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
            Vk.ACCESS_FRAGMENT_DENSITY_MAP_READ_BIT_EXT,
            Vk.ACCESS_HOST_READ_BIT,
            Vk.ACCESS_INDEX_READ_BIT,
            Vk.ACCESS_INDIRECT_COMMAND_READ_BIT,
            Vk.ACCESS_INPUT_ATTACHMENT_READ_BIT,
            Vk.ACCESS_SHADER_READ_BIT,
            Vk.ACCESS_SHADING_RATE_IMAGE_READ_BIT_NV,
            Vk.ACCESS_TRANSFER_READ_BIT,
            Vk.ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT,
            Vk.ACCESS_UNIFORM_READ_BIT,
            Vk.ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
        )
        y in x && return true
    end
    if Vk.ACCESS_MEMORY_WRITE_BIT in x
        if Vk.ACCESS_MEMORY_WRITE_BIT ∉ y
            x &= ~Vk.ACCESS_MEMORY_WRITE_BIT
        end
        x |= |(
            Vk.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            Vk.ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            Vk.ACCESS_COMMAND_PREPROCESS_WRITE_BIT_NV,
            Vk.ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            Vk.ACCESS_HOST_WRITE_BIT,
            Vk.ACCESS_SHADER_WRITE_BIT,
            Vk.ACCESS_TRANSFER_WRITE_BIT,
            Vk.ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
            Vk.ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
        )
        y in x && return true
    end
    if Vk.ACCESS_SHADER_READ_BIT in x
        if Vk.ACCESS_SHADER_READ_BIT ∉ y
            x &= ~Vk.ACCESS_SHADER_READ_BIT
        end
        x |= Vk.ACCESS_UNIFORM_READ_BIT
        y in x && return true
    end
    false
end

function ResourceClass(type::ResourceType)
    RESOURCE_TYPE_TEXTURE in type && return RESOURCE_CLASS_IMAGE
    RESOURCE_TYPE_IMAGE in type && return RESOURCE_CLASS_IMAGE
    RESOURCE_TYPE_INPUT_ATTACHMENT in type && return RESOURCE_CLASS_ATTACHMENT
    RESOURCE_TYPE_COLOR_ATTACHMENT in type && return RESOURCE_CLASS_ATTACHMENT
    RESOURCE_TYPE_DEPTH_ATTACHMENT in type && return RESOURCE_CLASS_ATTACHMENT
    RESOURCE_TYPE_STENCIL_ATTACHMENT in type && return RESOURCE_CLASS_ATTACHMENT
    RESOURCE_TYPE_BUFFER in type && return RESOURCE_CLASS_BUFFER
    RESOURCE_TYPE_VERTEX_BUFFER in type && return RESOURCE_CLASS_BUFFER
    RESOURCE_TYPE_INDEX_BUFFER in type && return RESOURCE_CLASS_BUFFER
    error("Resource type '$type' does not belong to any resource class.")
end
