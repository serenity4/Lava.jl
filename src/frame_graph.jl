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

function ImageResourceInfo(
  format;
  dims = [1.0, 1.0],
  size_unit = SIZE_SWAPCHAIN_RELATIVE,
  usage = Vk.ImageUsageFlag(0),
  aspect = Vk.ImageAspectFlag(0),
)
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
  frame::Frame
end

device(fg::FrameGraph) = fg.device

attribute(rg::AbstractMetaGraph, i, key) = get_prop(rg, i, key)
pass_attribute(rg::AbstractMetaGraph, i::Integer, key::Symbol) = attribute(rg, i, key)
pass_attribute(fg::FrameGraph, name::Symbol, key::Symbol) = attribute(fg.resource_graph, fg.passes[name], key)
resource_attribute(rg::AbstractMetaGraph, i::Integer, key::Symbol) = attribute(rg, i, key)
resource_attribute(fg::FrameGraph, name::Symbol, key::Symbol) = attribute(fg.resource_graph, fg.resources[name], key)
function set_attribute(fg::FrameGraph, pass::Symbol, resource::Symbol, key::Symbol, value)
  set_prop!(fg.resource_graph, fg.passes[pass], fg.resources[resource], key, value)
end

virtual_image(g, idx) = resource_attribute(g, idx, :virtual_resource)::ImageResourceInfo
virtual_buffer(g, idx) = resource_attribute(g, idx, :virtual_resource)::BufferResourceInfo
virtual_attachment(g, idx) = resource_attribute(g, idx, :virtual_resource)::AttachmentResourceInfo

image(g, idx) = resource_attribute(g, idx, :physical_resource)::Image
buffer(g::Union{FrameGraph,MetaGraph}, idx) = resource_attribute(g, idx, :physical_resource)::Buffer
attachment(g, idx) = resource_attribute(g, idx, :physical_resource)::Attachment

resource_type(g, i, j) = attribute(g, Edge(i, j), :type)::ResourceType
access(g, i, j) = attribute(g, Edge(i, j), :access)::MemoryAccess
access_bits(g, i, j) = attribute(g, Edge(i, j), :access_bits)::Vk.AccessFlag
image_layout(g, i, j) = attribute(g, Edge(i, j), :image_layout)::Vk.ImageLayout
clear_value(g, i, j) = attribute(g, Edge(i, j), :clear_value)::Vk.ClearValue
aspect(g, i, j) = attribute(g, Edge(i, j), :aspect)::Vk.ImageAspectFlag
# currently unused (not exposed to the user)
stages(g, i, j) = attribute(g, Edge(i, j), :stages)::Vk.PipelineStageFlag

resource_class(g, idx) = resource_attribute(g, idx, :class)::ResourceClass
current_layout(g, idx) = resource_attribute(g, idx, :current_layout)::Vk.ImageLayout
format(g, idx) = resource_attribute(g, idx, :format)::Vk.Format
buffer_usage(g, idx) = resource_attribute(g, idx, :usage)::Vk.BufferUsageFlag
image_usage(g, idx) = resource_attribute(g, idx, :usage)::Vk.ImageUsageFlag
_size(g, idx) = resource_attribute(g, idx, :size)::Int
_dims(g, idx) = resource_attribute(g, idx, :dims)::Vector{Int}
name(g, idx) = attribute(g, idx, :name)::Symbol
last_write(g, idx) = resource_attribute(g, idx, :last_write)::Pair{Vk.AccessFlag,Vk.PipelineStageFlag}
synchronization_state(g, idx) = resource_attribute(g, idx, :synchronization_state)::Dictionary{Vk.AccessFlag,Vk.PipelineStageFlag}
access(g, idx) = resource_attribute(g, idx, :access)::MemoryAccess
stages(g, idx) = pass_attribute(g, idx, :stages)::Vk.PipelineStageFlag
render_function(g, idx) = pass_attribute(g, idx, :render_function)
render_pass(g, idx) = pass_attribute(g, idx, :pass)::RenderPass

function FrameGraph(device::Device, frame::Frame = Frame(device))
  FrameGraph(device, MetaGraph(), Dictionary(), Dictionary(), frame)
end

function add_pass!(render_function, fg::FrameGraph, name::Symbol, pass::RenderPass, stages::Vk.PipelineStageFlag = Vk.PIPELINE_STAGE_ALL_GRAPHICS_BIT)
  !haskey(fg.passes, name) || error("Pass '$name' was already added to the frame graph. Passes can only be provided once.")
  g = fg.resource_graph
  add_vertex!(g)
  v = nv(g)
  set_prop!(g, v, :name, name)
  set_prop!(g, v, :stages, stages)
  set_prop!(g, v, :render_function, render_function)
  set_prop!(g, v, :pass, pass)
  insert!(fg.passes, name, v)
  nothing
end

add_pass!(fg::FrameGraph, pass::Pass) = add_pass!(fg, pass.name, pass.stages)

function add_resource!(fg::FrameGraph, name::Symbol, info::ResourceInfo)
  !haskey(fg.resources, name) || error("Resource '$name' was already added to the frame graph. Resources can only be provided once.")
  g = fg.resource_graph
  add_vertex!(g)
  v = nv(g)
  set_prop!(g, v, :name, name)
  set_prop!(g, v, :access, MemoryAccess(0))
  insert!(fg.resources, name, v)
  if info isa BufferResourceInfo
    set_prop!(g, v, :size, info.size)
    set_prop!(g, v, :usage, info.usage)
    set_prop!(g, v, :class, RESOURCE_CLASS_BUFFER)
  else
    image_info = if info isa AttachmentResourceInfo
      set_prop!(g, v, :class, RESOURCE_CLASS_ATTACHMENT)
      info.image_info
    else
      set_prop!(g, v, :class, RESOURCE_CLASS_IMAGE)
      info
    end
    set_prop!(g, v, :dims, image_info.dims)
    set_prop!(g, v, :usage, image_info.usage)
    set_prop!(g, v, :aspect, image_info.aspect)
    set_prop!(g, v, :format, image_info.format)
  end
  if haskey(fg.frame.resources, name)
    set_prop!(g, v, :physical_resource, fg.frame.resources[name].data)
    if resource_class(g, v) in (RESOURCE_CLASS_IMAGE, RESOURCE_CLASS_ATTACHMENT)
      set_prop!(g, v, :current_layout, image_layout(fg.frame.resources[name].data))
    end
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
    add_edge!(g, i, v) || error(
      "A resource usage between pass '$pass' and resource '$name' was already added to the frame graph. Resource usages can only be provided once.",
    )
    set_prop!(g, i, v, :type, usage.type)
    set_prop!(g, i, v, :access, usage.access)
  end
end

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
      !haskey(dict, name) || error("Resource $name for pass $f specified multiple times in read access")
      insert!(dict, name, ResourceUsage(type, READ))
    end
    for (name, type) in extract_resource_spec.(writes)
      usage = if haskey(dict, name)
        WRITE ∉ dict[name].access || error("Resource $name for pass $f specified multiple times in write access")
        ResourceUsage(type | dict[name].type, WRITE | READ)
      else
        ResourceUsage(type, WRITE)
      end
      set!(dict, name, usage)
    end
    !haskey(usages, f) || error("Pass $f is specified more than once")
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
    :($r::Buffer) => (r => RESOURCE_TYPE_BUFFER)
    :($r::Color) => (r => RESOURCE_TYPE_COLOR_ATTACHMENT)
    :($r::Depth) => (r => RESOURCE_TYPE_DEPTH_ATTACHMENT)
    :($r::Stencil) => (r => RESOURCE_TYPE_STENCIL_ATTACHMENT)
    :($r::Depth::Stencil) || :($_::Stencil::Depth) => (r => RESOURCE_TYPE_DEPTH_ATTACHMENT | RESOURCE_TYPE_STENCIL_ATTACHMENT)
    :($r::Texture) => (r => RESOURCE_TYPE_TEXTURE)
    :($r::Image::Storage) => (r => RESOURCE_TYPE_IMAGE | RESOURCE_TYPE_STORAGE)
    :($r::Input) => (r => RESOURCE_TYPE_INPUT_ATTACHMENT)
    ::Symbol => error("Resource type annotation required for $ex")
    _ => error("Invalid or unsupported resource type annotation for $ex")
  end
end

function extract_resource_name(ex::Expr)
  @match ex begin
    :($r::$_::$_) => r
    :($r::$_) => r
    _ => error("Cannot extract resource name from $ex")
  end
end

function clear_attachments(fg::FrameGraph, pass::Symbol, color_clears, depth_clears = [], stencil_clears = [])
  clears = Dictionary{Symbol,Vk.ClearValue}()
  for (resource, color) in color_clears
    clear_value = Vk.ClearValue(Vk.ClearColorValue(convert(NTuple{4,Float32}, color)))
    insert!(clears, resource, clear_value)
  end
  for (resource, depth) in depth_clears
    clear_value = Vk.ClearValue(Vk.ClearDepthStencilValue(depth, 0))
    insert!(clears, resource, clear_value)
  end
  for (resource, stencil) in stencil_clears
    clear_value = Vk.ClearValue(Vk.ClearDepthStencilValue(0.0, stencil))
    insert!(clears, resource, clear_value)
  end
  for (resource, clear_value) in pairs(clears)
    set_attribute(fg, pass, resource, :clear_value, clear_value)
  end
end

function execution_graph(fg::FrameGraph)
  g = fg.resource_graph
  eg = SimpleDiGraph(length(fg.passes))
  for pass in fg.passes
    resources = neighbors(g, pass)
    for resource in resources
      WRITE in access(g, pass, resource) || continue
      for dependent_pass in neighbors(g, resource)
        dependent_pass == pass && continue
        add_edge!(eg, dependent_pass, pass)
      end
    end
  end
  eg
end

"""
Submit rendering commands to a device.

A command buffer is recorded, which may be split into multiple ones to take advantage of multithreading,
and is then submitted them to the provided device. A custom primary command buffer can be optionally passed as a keyword,
mostly intended for debugging purposes.

A semaphore to wait for can be provided to synchronize with other commands.
"""
function render(fg::FrameGraph; semaphore = nothing, command_buffer = nothing, submit = true)
  prepare!(fg)
  passes = sort_passes(fg)
  for pass in passes
    # create render pass & framebuffer objects
    prepare_render_pass(fg, pass)
  end
  records, pipeline_hashes = record_commands!(fg, passes)
  !isempty(pipeline_hashes) || error("No draw calls detected.")
  device = Lava.device(fg)
  create_pipelines!(device)

  # fill command buffer with synchronization commands & recorded commands
  if isnothing(command_buffer)
    command_buffer = request_command_buffer(device, Vk.QUEUE_GRAPHICS_BIT)
  end
  first_pipeline = device.pipeline_ht[first(values(pipeline_hashes))]
  initialize(command_buffer, device, fg.frame.gd, first_pipeline)
  flush(command_buffer, fg, passes, records, pipeline_hashes)
  submit || return command_buffer

  # submit rendering work
  wait_semaphores = device.transfer_ops
  !isnothing(semaphore) && push!(wait_semaphores, semaphore)
  submit_info = Vk.SubmitInfo2KHR(wait_semaphores, [Vk.CommandBufferSubmitInfoKHR(command_buffer)], [])
  Lava.submit(device, command_buffer.queue_family_index, [submit_info]; signal_fence = true, release_after_completion = [Ref(fg)])
end

function prepare!(fg::FrameGraph)
  resolve_attributes!(fg)
  create_physical_resources!(fg)
end

function record_commands!(fg::FrameGraph, passes)
  records = CompactRecord[]
  pipeline_hashes = Dictionary{ProgramInstance,UInt64}()
  g = fg.resource_graph

  # record commands and submit pipelines for creation
  for pass in passes
    record = CompactRecord(fg, pass)
    f = render_function(g, pass)
    f(record)
    push!(records, record)
    merge!(pipeline_hashes, submit_pipelines!(device(fg), render_pass(g, pass), record))
  end

  records, pipeline_hashes
end

function Base.flush(cb::CommandBuffer, fg::FrameGraph, passes, records, pipeline_hashes)
  binding_state = BindState()
  for (pass, record) in zip(passes, records)
    synchronize_before(cb, fg, pass)
    begin_render_pass(cb, fg, pass)
    binding_state = flush(cb, record, device(fg), binding_state, pipeline_hashes)
    Vk.cmd_end_render_pass_2(cb, Vk.SubpassEndInfo())
    synchronize_after(cb, fg, pass)
  end
end

function sort_passes(fg::FrameGraph)
  indices = topological_sort_by_dfs(execution_graph(fg))
  collect(values(fg.passes))[indices]
end

"""
Build barriers for all resources that require it.

Requires the extension `VK_KHR_synchronization2`.
"""
function synchronize_before(cb, fg::FrameGraph, pass::Integer)
  g = fg.resource_graph
  deps = Vk.DependencyInfoKHR([], [], [])
  for resource in neighbors(g, pass)
    type = resource_type(g, pass, resource)
    class = resource_class(g, resource)
    (req_access, req_stages) = access(g, pass, resource) => stages(g, pass)
    # if the resource was not written to recently, no synchronization is required
    if has_prop(g, resource, :last_write)
      (r_access, p_stages) = last_write(g, resource)
      req_access_bits = access_bits(type, req_access, req_stages)
      sync_state = synchronization_state(g, resource)
      synced_stages = Vk.PipelineStageFlag(0)
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
        @switch class begin
          @case &RESOURCE_CLASS_BUFFER
          buff = buffer(g, resource)
          barrier = Vk.BufferMemoryBarrier2KHR(p_stages, req_stages, r_access, req_access_bits, 0, 0, buff, offset(buff), size(buff))
          push!(deps.buffer_memory_barriers, barrier)
          @case &RESOURCE_CLASS_IMAGE || &RESOURCE_CLASS_ATTACHMENT
          view = if class == RESOURCE_CLASS_IMAGE
            View(image(g, resource))
          else
            attachment(g, resource).view
          end
          new_layout = image_layout(g, pass, resource)
          range = subresource_range(view)
          barrier = Vk.ImageMemoryBarrier2KHR(
            C_NULL,
            p_stages,
            req_stages,
            r_access,
            req_access_bits,
            current_layout(g, resource),
            new_layout,
            0,
            0,
            view.image,
            range,
          )
          push!(deps.image_memory_barriers, barrier)
          set_prop!(g, resource, :current_layout, new_layout)
          view.image.layout[] = new_layout
        end
        set!(sync_state, req_access_bits, req_stages)
      end
    elseif class in (RESOURCE_CLASS_IMAGE, RESOURCE_CLASS_ATTACHMENT) && current_layout(g, resource) ≠ image_layout(g, pass, resource)
      # perform the required layout transition without further synchronization
      view = if class == RESOURCE_CLASS_IMAGE
        View(image(g, resource))
      else
        attachment(g, resource).view
      end
      new_layout = image_layout(g, pass, resource)
      barrier = Vk.ImageMemoryBarrier2KHR(
        C_NULL,
        Vk.PipelineStageFlag(0),
        Vk.PipelineStageFlag(0),
        Vk.AccessFlag(0),
        Vk.AccessFlag(0),
        current_layout(g, resource),
        new_layout,
        0,
        0,
        view.image,
        subresource_range(view),
      )
      push!(deps.image_memory_barriers, barrier)
      set_prop!(g, resource, :current_layout, new_layout)
      view.image.layout[] = new_layout
    end
    if WRITE in req_access
      set_prop!(g, resource, :last_write, access_bits(resource_type(g, pass, resource), req_access, req_stages) => req_stages)
      set_prop!(g, resource, :synchronization_state, Dictionary{Vk.AccessFlag,Vk.PipelineStageFlag}())
    end
  end
  (!isempty(deps.memory_barriers) || !isempty(deps.image_memory_barriers) || !isempty(deps.buffer_memory_barriers)) &&
    Vk.cmd_pipeline_barrier_2_khr(cb, deps)
end

function prepare_render_pass(fg::FrameGraph, pass::Integer)
  g = fg.resource_graph
  attachment_nodes = (resource for resource in neighbors(g, pass) if ResourceClass(resource_type(g, pass, resource)) == RESOURCE_CLASS_ATTACHMENT)
  attachms = Attachment[]
  attach_descs = Vk.AttachmentDescription2[]
  clear_values = Vk.ClearValue[]

  subpass_descriptions = Vk.SubpassDescription2[]
  subpass_dependencies = Vk.SubpassDependency2[]

  for (i, node) in enumerate(attachment_nodes)
    # render pass (global) attachment info
    attachm = attachment(g, i)
    push!(attachms, attachm)
    clear = has_prop(g, pass, node, :clear_value)
    push!(attach_descs, Vk.AttachmentDescription2(attachm, clear, image_layout(g, pass, node), image_layout(g, pass, node), aspect(g, pass, node)))
    clear_val = if clear
      clear_value(g, pass, node)
    else
      # need a filler clear value
      Vk.ClearValue(Vk.ClearColorValue(ntuple(Returns(0.0f0), 4)))
    end
    push!(clear_values, clear_val)

    # subpass (local) attachment info
    # only one subpass per pass is currently supported
    # we could later look up an internal subpass graph to support more
    for subpass = 1:1
      stage_flags = stages(g, pass)
      pipeline_bind_point = Vk.PIPELINE_STAGE_COMPUTE_SHADER_BIT in stage_flags ? Vk.PIPELINE_BIND_POINT_COMPUTE : Vk.PIPELINE_BIND_POINT_GRAPHICS
      color_attachments = Vk.AttachmentReference2[]
      depth_stencil_attachment = C_NULL
      input_attachments = Vk.AttachmentReference2[]
      type = resource_type(g, pass, node)
      ref = Vk.AttachmentReference2(i - 1, image_layout(g, pass, node), aspect(g, pass, node))
      if RESOURCE_TYPE_COLOR_ATTACHMENT in type
        push!(color_attachments, ref)
      end
      if RESOURCE_TYPE_DEPTH_ATTACHMENT in type || RESOURCE_TYPE_STENCIL_ATTACHMENT in type
        depth_stencil_attachment = ref
      end
      if RESOURCE_TYPE_INPUT_ATTACHMENT in type
        push!(input_attachments, ref)
      end
      push!(subpass_descriptions, Vk.SubpassDescription2(pipeline_bind_point, 0, input_attachments, color_attachments, []; depth_stencil_attachment))

      acc_bits = access_bits(g, pass, node)
      push!(
        subpass_dependencies,
        Vk.SubpassDependency2(Vk.SUBPASS_EXTERNAL, subpass - 1, 0; dst_stage_mask = stage_flags, dst_access_mask = acc_bits),
      )
    end
  end
  rp = render_pass(g, pass)
  rp_handle = Vk.RenderPass(fg.device, attach_descs, subpass_descriptions, subpass_dependencies, [])
  (; width, height) = rp.area.extent
  fb = Vk.Framebuffer(fg.device, rp_handle, [att.view for att in attachms], width, height, 1)
  set_prop!(g, pass, :render_pass_handle, rp_handle)
  set_prop!(g, pass, :framebuffer, fb)
  set_prop!(g, pass, :clear_values, clear_values)
end

function begin_render_pass(cb, fg::FrameGraph, pass::Integer)
  g = fg.resource_graph
  Vk.cmd_begin_render_pass_2(cb,
    Vk.RenderPassBeginInfo(
      pass_attribute(g, pass, :render_pass_handle),
      pass_attribute(g, pass, :framebuffer),
      render_pass(g, pass).area,
      pass_attribute(g, pass, :clear_values),
    ),
    Vk.SubpassBeginInfo(Vk.SUBPASS_CONTENTS_INLINE),
  )
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

function resolve_attributes!(fg::FrameGraph)
  g = fg.resource_graph
  for pass in fg.passes
    for resource in neighbors(g, pass)
      type = resource_type(g, pass, resource)
      resource_access = access(g, pass, resource)
      set_prop!(g, resource, :access, access(g, resource) | resource_access)
      pipeline_stages = if has_prop(g, pass, resource, :stages)
        stages(g, pass, resource)
      else
        stages(g, pass)
      end
      # class-independent attributes
      acc_bits = access_bits(type, resource_access, pipeline_stages)
      set_prop!(g, pass, resource, :access_bits, acc_bits)

      # class-dependent attributes
      class = resource_class(g, resource)
      if class == RESOURCE_CLASS_IMAGE || class == RESOURCE_CLASS_ATTACHMENT
        usage = image_usage_bits(type, resource_access)
        set_prop!(g, pass, resource, :usage, usage)
        layout = image_layout(type, resource_access, pipeline_stages)
        set_prop!(g, pass, resource, :image_layout, layout)
        aspect = aspect_bits(type)
        set_prop!(g, pass, resource, :aspect, aspect)
        set_prop!(g, resource, :usage, usage | image_usage(g, resource))
      else
        usage = buffer_usage_bits(type, resource_access)
        set_prop!(g, pass, resource, :usage, usage)
        set_prop!(g, resource, :usage, usage | buffer_usage(g, resource))
      end
    end
  end
end

function create_physical_resources!(fg::FrameGraph)
  g = fg.resource_graph
  for resource in fg.resources |> Filter(x -> !has_prop(g, x, :physical_resource))
    c = resource_class(g, resource)
    if c == RESOURCE_CLASS_IMAGE || c == RESOURCE_CLASS_ATTACHMENT
      image = ImageBlock(device(fg), Tuple(_dims(g, resource)), format(g, resource), image_usage(g, resource))
      allocate!(image, MEMORY_DOMAIN_DEVICE)
      if c == RESOURCE_CLASS_ATTACHMENT
        set_prop!(g, resource, :physical_resource, Attachment(View(image), access(g, resource)))
      else
        set_prop!(g, resource, :physical_resource, image)
      end
    else
      buffer = BufferBlock(device(fg), _size(g, resource); usage = buffer_usage(g, resource))
      allocate!(buffer, MEMORY_DOMAIN_DEVICE)
      set_prop!(g, resource, :physical_resource, buffer)
    end
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
