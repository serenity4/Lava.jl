struct Device <: LavaAbstraction
    handle::Vk.Device
    extensions::Vector{String}
    features::Vk.PhysicalDeviceFeatures2
    queues::QueueDispatch
    pipeline_ht::HashTable{Pipeline}
    pending_pipelines::Vector{Vk.GraphicsPipelineCreateInfo}
    shader_cache::ShaderCache
    transfer_ops::Vector{Vk.SemaphoreSubmitInfoKHR}
    command_pools::CommandPools
end

vk_handle_type(::Type{Device}) = Vk.Device

function Device(physical_device::Vk.PhysicalDevice, extensions, queue_config, features::Vk.PhysicalDeviceFeatures2; surface = nothing, next = C_NULL)
    infos = queue_infos(QueueDispatch, physical_device, queue_config)
    info = Vk.DeviceCreateInfo(
        infos,
        [],
        extensions;
        next,
    )

    handle = unwrap(create(Device, physical_device, info))
    queues = QueueDispatch(handle, infos; surface)
    Device(handle, extensions, features, queues, HashTable{Pipeline}(), [], ShaderCache(handle), [], CommandPools(handle))
end

function request_command_buffer(device::Device, usage::Vk.QueueFlag)
    index = get_queue_family(device.queues, usage)
    pool = request_pool!(device.command_pools, index)
    handle = first(unwrap(Vk.allocate_command_buffers(device, Vk.CommandBufferAllocateInfo(pool, Vk.COMMAND_BUFFER_LEVEL_PRIMARY, 1))))
    CommandBuffer(handle, index)
end

queue_family_indices(device::Device) = queue_family_indices(device.queues)

submit(device, args...; kwargs...) = submit(device.queues, args...; kwargs...)
