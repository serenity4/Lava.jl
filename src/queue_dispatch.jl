struct Queue <: LavaAbstraction
    handle::Vk.Queue
    capabilities::Vk.QueueFlag
    index::UInt8
    family::UInt8
end

vk_handle_type(::Type{Queue}) = Vk.Queue

struct QueueDispatch
    queues::Dictionary{Vk.QueueFlag,Vector{Queue}}
    present_queue::Optional{Queue}
    """
    Build a `QueueDispatch` structure from a given device and configuration.
    If a surface is provided, then a queue that supports presentation on this surface will be filled in `present_queue`.

    !!! warning
        `device` must have been created with a consistent number of queues as requested in the provided queue configuration.
        It is highly recommended to have created the device with the result of `queue_infos(QueueDispatch, physical_device, config)`.
    """
    function QueueDispatch(device, config; surface = nothing)
        pdevice = physical_device(device)
        families = Vk.find_queue_family.(pdevice, collect(keys(config)))
        queues = dictionary(map(zip(keys(config), families)) do (capabilities, family)
            capabilities => map(0:config[capabilities] - 1) do index
                info = Vk.DeviceQueueInfo2(family, index)
                Queue(Vk.get_device_queue_2(device, info), capabilities, index + 1, family + 1)
            end
        end)
        present_queue = if !isnothing(surface)
            idx = findfirst(families) do family
                unwrap(Vk.get_physical_device_surface_support_khr(pdevice, family, surface))
            end
            if isnothing(idx)
                error("Could not find a queue that supports presentation on the provided surface.")
            else
                first(values(queues)[idx])
            end
        else
            nothing
        end
        new(queues, present_queue)
    end
end

function queue_infos(::Type{QueueDispatch}, physical_device::Vk.PhysicalDevice, config)
    all(==(1), config) || error("Only one queue per property is currently supported")
    families = Vk.find_queue_family.(physical_device, collect(keys(config)))
    Vk.DeviceQueueCreateInfo.(families, ones.(collect(config)))
end

function submit(dispatch::QueueDispatch, properties::Vk.QueueFlag, submit_infos; fence = C_NULL)
    q = queue(dispatch, properties)
    unwrap(Vk.queue_submit_2_khr(q, submit_infos; fence))
    q
end

function queue(dispatch::QueueDispatch, properties::Vk.QueueFlag)
    if properties in keys(dispatch.queues)
        first(dispatch.queues[properties])
    else
        for props in keys(dispatch.queues)
            if properties in props
                return first(dispatch.queues[props])
            end
        end
        error("Could not find a queue matching with the required properties $properties.")
    end
end

function present(dispatch::QueueDispatch, present_info::Vk.PresentInfoKHR)
    queue = dispatch.present_queue
    if isnothing(queue)
        error("No presentation queue was specified for $dispatch")
    else
        Vk.queue_present_khr(queue, present_info)
    end
end

function queue_family_indices(dispatch::QueueDispatch; include_present = true)
    indices = map(dispatch.queues) do queues
        map(Base.Fix2(getproperty, :family), queues)
    end
    indices = reduce(vcat, indices)
    if include_present && !isnothing(dispatch.present_queue)
        push!(indices, dispatch.present_queue.family)
    end
    sort!(unique!(indices))
end
