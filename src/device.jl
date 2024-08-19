struct Device <: LavaAbstraction
  handle::Vk.Device
  api_version::VersionNumber
  extensions::Vector{String}
  features::Vk.PhysicalDeviceFeatures2
  queues::QueueDispatch
  pipeline_ht_graphics::HashTable{Pipeline}
  pipeline_ht_compute::HashTable{Pipeline}
  pipeline_layout_ht::HashTable{PipelineLayout}
  pipeline_layouts::Dictionary{Vk.PipelineLayout,PipelineLayout}
  pending_pipelines_graphics::Vector{Vk.GraphicsPipelineCreateInfo}
  pending_pipelines_compute::Vector{Vk.ComputePipelineCreateInfo}
  shader_cache::ShaderCache
  transfer_ops::Vector{Vk.SemaphoreSubmitInfo}
  command_pools::CommandPools
  spirv_features::SupportedFeatures
  fence_pool::FencePool
  descriptors::GlobalDescriptors
  alignment::VulkanAlignment
end

vk_handle_type(::Type{Device}) = Vk.Device

function Device(physical_device::Vk.PhysicalDevice, application_version::VersionNumber, extensions, queue_config,
  features::Vk.PhysicalDeviceFeatures2; next = C_NULL)

  infos = queue_infos(QueueDispatch, physical_device, queue_config)
  info = Vk.DeviceCreateInfo(
    infos,
    [],
    extensions;
    next,
  )

  supported_device_version = Vk.get_physical_device_properties_2(physical_device).properties.api_version
  api_version = min(application_version, supported_device_version)

  handle = unwrap(create(Device, physical_device, info))
  queues = QueueDispatch(handle, infos)
  alignment = VulkanAlignment()
  Device(
    handle,
    api_version,
    extensions,
    features,
    queues,
    HashTable{Pipeline}(),
    HashTable{Pipeline}(),
    HashTable{PipelineLayout}(),
    Dictionary(),
    Vk.GraphicsPipelineCreateInfo[],
    Vk.ComputePipelineCreateInfo[],
    ShaderCache(handle),
    [],
    CommandPools(handle),
    SupportedFeatures(physical_device, api_version, extensions, features),
    FencePool(handle),
    GlobalDescriptors(handle),
    alignment,
  )
end

Base.wait(device::Device) = Vk.device_wait_idle(device)

function Base.empty!(device::Device)
  unwrap(wait(device))
  empty!(device.pipeline_ht_graphics)
  empty!(device.pipeline_ht_compute)
  empty!(device.pipeline_layout_ht)
  empty!(device.pipeline_layouts)
  empty!(device.pending_pipelines_graphics)
  empty!(device.pending_pipelines_compute)
  empty!(device.shader_cache)
  empty!(device.transfer_ops)
  empty!(device.fence_pool)
  empty!(device.descriptors)
  nothing
end

Shader(device::Device, source::ShaderSource) = Shader(device.shader_cache, source)
ShaderSource(device::Device, info::ShaderInfo) = ShaderSource(device.shader_cache, info)

const QUEUE_GENERAL_BITS = Vk.QUEUE_GRAPHICS_BIT | Vk.QUEUE_COMPUTE_BIT | Vk.QUEUE_TRANSFER_BIT

function request_command_buffer(device::Device, queue_usage_bits::Vk.QueueFlag = QUEUE_GENERAL_BITS)
  index = get_queue_family(device.queues, queue_usage_bits)
  pool = request_pool!(device.command_pools, index)
  handle = first(unwrap(Vk.allocate_command_buffers(device, Vk.CommandBufferAllocateInfo(pool, Vk.COMMAND_BUFFER_LEVEL_PRIMARY, 1))))
  cb = SimpleCommandBuffer(handle, index, device.queues)
  push!(cb.to_free, cb)
  start_recording(cb)
  cb
end

queue_family_indices(device::Device) = queue_family_indices(device.queues)

submit(device::Device, args...; kwargs...) = submit(device.queues, args...; kwargs...)

"Split a vector in `n` equivalent chunks."
function split_vec(vec, n)
  nv = length(vec)
  tsize = cld(n, nv)
  ranges = [(1 + tsize * i):(min(1 + tsize * (i + 1), nv)) for i in 0:(n - 1)]
  [vec[range] for range in ranges]
end

function pmap(f, collection, init)
  tasks = Task[]
  # Initialize a per-thread storage for results.
  res = fill(init, length(collection))
  # Spawn all the tasks.
  for (i, el) in enumerate(collection)
    t = Threads.@spawn begin
      res[i] = f(el)
    end
    push!(tasks, t)
  end
  # Wait for completion.
  for t in tasks
    wait(t)
  end
  res
end

function create_graphics_pipelines!(device::Device, infos)
  # Assume that each available thread will be able to create a set of pipelines in batch mode.
  # We don't create individual pipelines for performance reasons as the implementation is
  # likely to setup internal mutexes for each batch which allow pipeline creation to be concurrent.

  #FIXME: This segfaults at second try.
  # infos_vec = split_vec(infos, Threads.nthreads())
  # handles_vec = pmap(infos_vec, Vk.Pipeline[]) do infos
  #   isempty(infos) && return Vk.Pipeline[]
  #   first(unwrap(Vk.create_graphics_pipelines(device, infos)))
  # end
  # handles = reduce(vcat, handles_vec)
  handles = first(unwrap(Vk.create_graphics_pipelines(device, infos)))

  map((x, y) -> graphics_pipeline(device, x, y), handles, infos)
end

function create_compute_pipelines!(device::Device, infos)
  # Assume that each available thread will be able to create a set of pipelines in batch mode.
  # We don't create individual pipelines for performance reasons as the implementation is
  # likely to setup internal mutexes for each batch which allow pipeline creation to be concurrent.

  #FIXME: This segfaults at second try.
  # infos_vec = split_vec(infos, Threads.nthreads())
  # handles_vec = pmap(infos_vec, Vk.Pipeline[]) do infos
  #   isempty(infos) && return Vk.Pipeline[]
  #   first(unwrap(Vk.create_compute_pipelines(device, infos)))
  # end
  # handles = reduce(vcat, handles_vec)
  handles = first(unwrap(Vk.create_compute_pipelines(device, infos)))

  map((x, y) -> compute_pipeline(device, x, y), handles, infos)
end

graphics_pipeline(device::Device, handle::Vk.Pipeline, info::Vk.GraphicsPipelineCreateInfo) =
  Pipeline(handle, PipelineType(Vk.PIPELINE_BIND_POINT_GRAPHICS), pipeline_layout(device, info.layout))
compute_pipeline(device::Device, handle::Vk.Pipeline, info::Vk.ComputePipelineCreateInfo) =
  Pipeline(handle, PipelineType(Vk.PIPELINE_BIND_POINT_COMPUTE), pipeline_layout(device, info.layout))

function create_pipelines!(device::Device)
  batch_create!(infos -> create_graphics_pipelines!(device, infos), device.pipeline_ht_graphics, device.pending_pipelines_graphics)
  batch_create!(infos -> create_compute_pipelines!(device, infos), device.pipeline_ht_compute, device.pending_pipelines_compute)
  empty!(device.pending_pipelines_graphics)
  empty!(device.pending_pipelines_compute)
  nothing
end

function pipeline_layout(device::Device)
  info = Vk.PipelineLayoutCreateInfo(
    [device.descriptors.gset.layout],
    [Vk.PushConstantRange(Vk.SHADER_STAGE_ALL, 0, sizeof(DeviceAddressBlock))],
  )
  get!(device.pipeline_layout_ht, info) do info
    handle = unwrap(Vk.create_pipeline_layout(device, info))
    layout = PipelineLayout(handle, info.set_layouts, info.push_constant_ranges)
    insert!(device.pipeline_layouts, handle, layout)
    layout
  end
end

pipeline_layout(device::Device, handle::Vk.PipelineLayout) = device.pipeline_layouts[handle]

function request_pipeline(device::Device, info::Vk.GraphicsPipelineCreateInfo)
  push!(device.pending_pipelines_graphics, info)
  hash(info)
end

function request_pipeline(device::Device, info::Vk.ComputePipelineCreateInfo)
  push!(device.pending_pipelines_compute, info)
  hash(info)
end

@forward_methods Device field = :fence_pool fence
@forward_methods Device field = :queues set_presentation_queue

function Base.show(io::IO, device::Device)
  print(io, Device, "($(device.handle))")
end

buffer_resource(size::Integer; name = nothing) = Resource(LogicalBuffer(size); name)

function buffer_resource(device::Device, data; name = nothing, memory_domain::MemoryDomain = MEMORY_DOMAIN_DEVICE, usage_flags = Vk.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, submission = nothing, queue_family_indices = queue_family_indices(device), sharing_mode = Vk.SHARING_MODE_EXCLUSIVE)
  buffer = Buffer(device; data, memory_domain, usage_flags, submission, queue_family_indices, sharing_mode)
  Resource(buffer; name)
end

image_resource(format::Union{Vk.Format, DataType}, dims; name = nothing, mip_levels = 1, layers = 1, samples = nothing) = Resource(LogicalImage(format, dims; mip_levels, layers, samples); name)

function image_resource(device::Device, data;
  name = nothing,
  flags = nothing,
  format = nothing,
  memory_domain = MEMORY_DOMAIN_DEVICE,
  optimal_tiling = true,
  usage_flags = Vk.IMAGE_USAGE_SAMPLED_BIT,
  dims = nothing,
  samples = 1,
  queue_family_indices = queue_family_indices(device),
  sharing_mode = Vk.SHARING_MODE_EXCLUSIVE,
  mip_levels = 1,
  layers = 1,
  layout::Optional{Vk.ImageLayout} = nothing,
  submission = isnothing(data) ? nothing : SubmissionInfo(signal_fence = fence(device)))

  image = Image(device; data, flags, format, memory_domain, optimal_tiling, usage_flags, dims, samples, queue_family_indices, sharing_mode, mip_levels, layers, layout, submission)
  Resource(image; name)
end

function attachment_resource(format::Union{Vk.Format, DataType}, dims = nothing; name = nothing, kwargs...)
  Resource(LogicalAttachment(format, dims; kwargs...); name)
end

function attachment_resource(device::Device, data; name = nothing, access::MemoryAccess = READ | WRITE, kwargs...)
  Resource(Attachment(device, data; access, kwargs...); name)
end

attachment_resource(view::ImageView, access::MemoryAccess; name = nothing) = Resource(Attachment(view, access); name)
