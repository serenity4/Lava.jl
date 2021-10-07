struct Device <: LavaAbstraction
    handle::Vk.Device
    extensions::Vector{String}
    features::Vk.PhysicalDeviceFeatures2
    queues::QueueDispatch
    pipeline_ht::HashTable{Pipeline}
    pipeline_layout_ht::HashTable{PipelineLayout}
    pipeline_layouts::Dictionary{Vk.PipelineLayout,PipelineLayout}
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
    Device(handle, extensions, features, queues, HashTable{Pipeline}(), HashTable{PipelineLayout}(), Dictionary(), [], ShaderCache(handle), [], CommandPools(handle))
end

function request_command_buffer(device::Device, usage::Vk.QueueFlag)
    index = get_queue_family(device.queues, usage)
    pool = request_pool!(device.command_pools, index)
    handle = first(unwrap(Vk.allocate_command_buffers(device, Vk.CommandBufferAllocateInfo(pool, Vk.COMMAND_BUFFER_LEVEL_PRIMARY, 1))))
    cb = SimpleCommandBuffer(handle, index)
    start_recording(cb)
    cb
end

queue_family_indices(device::Device) = queue_family_indices(device.queues)

submit(device, args...; kwargs...) = submit(device.queues, args...; kwargs...)

function create_pipelines!(device::Device)
    batch_create!(device.pipeline_ht, device.pending_pipelines) do infos
        (handles, _) = unwrap(Vk.create_graphics_pipelines(device, infos))
        map(zip(handles, infos)) do (handle, info)
            Pipeline(handle, PipelineType(Vk.PIPELINE_BIND_POINT_GRAPHICS), pipeline_layout(device, info.layout))
        end
    end
end

function pipeline_layout(device::Device, resources)
    info = Vk.PipelineLayoutCreateInfo(
        [resources.gset.set.layout],
        [Vk.PushConstantRange(Vk.ShaderStageFlag(0x7fffffff), 0, sizeof(PushConstantData))],
    )
    get!(device.pipeline_layout_ht, info) do info
        handle = unwrap(Vk.create_pipeline_layout(device, info))
        layout = PipelineLayout(handle, info.set_layouts, info.push_constant_ranges)
        insert!(device.pipeline_layouts, handle, layout)
        layout
    end
end

pipeline_layout(device::Device, handle::Vk.PipelineLayout) = device.pipeline_layouts[handle]

function Base.show(io::IO, device::Device)
    print(io, Device, "($(device.handle))")
end
