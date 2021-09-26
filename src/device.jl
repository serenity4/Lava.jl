struct Device <: LavaAbstraction
    handle::Vk.Device
    extensions::Vector{String}
    features::Vk.PhysicalDeviceFeatures2
    queues::QueueDispatch
    pipeline_ht::HashTable{Pipeline}
    pending_pipelines::Vector{Vk.GraphicsPipelineCreateInfo}
    shader_cache::ShaderCache
end

vk_handle_type(::Type{Device}) = Vk.Device

function Device(physical_device::Vk.PhysicalDevice, extensions, queue_config, features::Vk.PhysicalDeviceFeatures2; surface = nothing, next = C_NULL)
    info = Vk.DeviceCreateInfo(
        queue_infos(QueueDispatch, physical_device, queue_config),
        [],
        extensions;
        next,
    )

    handle = unwrap(create(Device, physical_device, info))
    queues = QueueDispatch(handle, queue_config; surface)
    Device(handle, extensions, features, queues, HashTable{Pipeline}(), [], ShaderCache(handle))
end

queue_family_indices(device::Device) = queue_family_indices(device.queues)
