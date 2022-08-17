struct Device <: LavaAbstraction
  handle::Vk.Device
  api_version::VersionNumber
  extensions::Vector{String}
  features::Vk.PhysicalDeviceFeatures2
  queues::QueueDispatch
  pipeline_ht::HashTable{Pipeline}
  pipeline_layout_ht::HashTable{PipelineLayout}
  pipeline_layouts::Dictionary{Vk.PipelineLayout,PipelineLayout}
  pending_pipelines::Vector{Vk.GraphicsPipelineCreateInfo}
  shader_cache::ShaderCache
  transfer_ops::Vector{Vk.SemaphoreSubmitInfo}
  command_pools::CommandPools
  spirv_features::SupportedFeatures
  fence_pool::FencePool
  descriptors::GlobalDescriptors
  layout::VulkanLayout
end

vk_handle_type(::Type{Device}) = Vk.Device

function Device(physical_device::Vk.PhysicalDevice, application_version::VersionNumber, extensions, queue_config,
  features::Vk.PhysicalDeviceFeatures2; surface = nothing, next = C_NULL)

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
  queues = QueueDispatch(handle, infos; surface)
  Device(
    handle,
    api_version,
    extensions,
    features,
    queues,
    HashTable{Pipeline}(),
    HashTable{PipelineLayout}(),
    Dictionary(),
    [],
    ShaderCache(handle),
    [],
    CommandPools(handle),
    spirv_features(physical_device, api_version, extensions, features),
    FencePool(handle),
    GlobalDescriptors(handle),
    VulkanLayout(),
  )
end

const QUEUE_GENERAL_BITS = Vk.QUEUE_GRAPHICS_BIT | Vk.QUEUE_COMPUTE_BIT | Vk.QUEUE_TRANSFER_BIT

function request_command_buffer(device::Device, queue_usage_bits::Vk.QueueFlag = QUEUE_GENERAL_BITS)
  index = get_queue_family(device.queues, queue_usage_bits)
  pool = request_pool!(device.command_pools, index)
  handle = first(unwrap(Vk.allocate_command_buffers(device, Vk.CommandBufferAllocateInfo(pool, Vk.COMMAND_BUFFER_LEVEL_PRIMARY, 1))))
  # Inefficient, but will at least prevent memory leaks.
  finalizer(x -> Vk.free_command_buffers(x.command_pool.device, x.command_pool, [x]), handle)
  cb = SimpleCommandBuffer(handle, index, device.queues)
  push!(cb.to_free, cb)
  start_recording(cb)
  cb
end

queue_family_indices(device::Device) = queue_family_indices(device.queues)

submit(device, args...; kwargs...) = submit(device.queues, args...; kwargs...)

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

function create_pipelines(device::Device, infos)
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

graphics_pipeline(device::Device, handle::Vk.Pipeline, info::Vk.GraphicsPipelineCreateInfo) =
  Pipeline(handle, PipelineType(Vk.PIPELINE_BIND_POINT_GRAPHICS), pipeline_layout(device, info.layout))

function create_pipelines(device::Device)
  ret = batch_create!(Base.Fix1(create_pipelines, device), device.pipeline_ht, device.pending_pipelines)
  empty!(device.pending_pipelines)
  ret
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

@forward Device.fence_pool (fence,)
@forward Device.queues (set_presentation_queue,)

function Base.show(io::IO, device::Device)
  print(io, Device, "($(device.handle))")
end

buffer_resource(size::Integer) = logical_buffer(size)

function buffer_resource(device::Device, data; memory_domain::MemoryDomain = MEMORY_DOMAIN_DEVICE, usage_flags = Vk.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, submission = nothing, queue_family_indices = queue_family_indices(device), sharing_mode = Vk.SHARING_MODE_EXCLUSIVE)
  buffer = Buffer(device; data, memory_domain, usage_flags, submission, queue_family_indices, sharing_mode)
  Resource(RESOURCE_TYPE_BUFFER, buffer)
end

image_resource(format::Union{Vk.Format, DataType}, dims; mip_levels = 1, layers = 1) = logical_image(format, dims; mip_levels, layers)

function image_resource(device::Device, data;
  format = nothing,
  memory_domain = MEMORY_DOMAIN_DEVICE,
  optimal_tiling = true,
  usage_flags = Vk.IMAGE_USAGE_SAMPLED_BIT,
  dims = nothing,
  samples = 1,
  queue_family_indices = queue_family_indices(device),
  sharing_mode = Vk.SHARING_MODE_EXCLUSIVE,
  mip_levels = 1,
  array_layers = 1,
  layout::Optional{Vk.ImageLayout} = nothing,
  submission = isnothing(data) ? nothing : SubmissionInfo(signal_fence = fence(device)))

  image = Image(device; data, format, memory_domain, optimal_tiling, usage_flags, dims, samples, queue_family_indices, sharing_mode, mip_levels, array_layers, layout, submission)
  Resource(RESOURCE_TYPE_IMAGE, image)
end

function attachment_resource(format::Union{Vk.Format, DataType}, dims = nothing; kwargs...)
  logical_attachment(format, dims; kwargs...)
end

function attachment_resource(device::Device, data; access::MemoryAccess = READ | WRITE, aspect::Vk.ImageAspectFlag = Vk.IMAGE_ASPECT_COLOR_BIT, kwargs...)
  Resource(RESOURCE_TYPE_ATTACHMENT, Attachment(device, data; access, aspect, kwargs...))
end

attachment_resource(view::ImageView, access::MemoryAccess) = Resource(RESOURCE_TYPE_ATTACHMENT, Attachment(view, access))
