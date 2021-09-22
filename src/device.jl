struct Device <: LavaAbstraction
    handle::Vk.Device
    extensions::Vector{String}
    features::Vk.PhysicalDeviceFeatures
    queues::QueueDispatch
    pipeline_ht::HashTable{Pipeline}
    pending_pipelines::Vector{Vk.GraphicsPipelineCreateInfo}
    shader_cache::ShaderCache
end

vk_handle_type(::Type{Device}) = Vk.Device

function Device(physical_device::Vk.PhysicalDevice, extensions, queue_config; enabled_features = Vk.PhysicalDeviceFeatures(), surface = nothing)
    info = Vk.DeviceCreateInfo(
        queue_infos(QueueDispatch, physical_device, queue_config),
        [],
        extensions;
        enabled_features,
    )
    
    handle = unwrap(create(Device, physical_device, info))
    queues = QueueDispatch(handle, queue_config; surface)
    Device(handle, extensions, enabled_features, queues, HashTable{Pipeline}(), [], ShaderCache(handle))
end

queue_family_indices(device::Device) = queue_family_indices(device.queues)
