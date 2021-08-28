struct Instance <: LavaAbstraction
    handle::Vk.Instance
    layers::Vector{String}
    extensions::Vector{String}
    messenger::Optional{Vk.DebugUtilsMessengerEXT}
end

vk_handle_type(::Type{Instance}) = Vk.Instance

Instance(handle::Vk.Instance) = Instance(handle, [], [], nothing)

function Instance(layers, extensions, messenger_info::Optional{Vk.DebugUtilsMessengerCreateInfoEXT} = nothing; application_info = C_NULL)
    handle = unwrap(create(Instance, Vk.InstanceCreateInfo(layers, extensions; application_info, next = something(messenger_info, C_NULL))))
    if !isnothing(messenger_info)
        messenger = debug_messenger(handle, messenger_info)
    else
        messenger = nothing
    end
    Instance(handle, layers, extensions, messenger)
end

struct Device <: LavaAbstraction
    handle::Vk.Device
    extensions::Vector{String}
    features::Vk.PhysicalDeviceFeatures
    queues::QueueDispatch
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
    Device(handle, extensions, enabled_features, queues)
end

queue_family_indices(device::Device) = queue_family_indices(device.queues)

function init(;
    instance_layers = String[],
    instance_extensions = String[],
    application_info = Vk.ApplicationInfo(v"1", v"1", v"1.2"),
    device_extensions = String[],
    enabled_features = Vk.PhysicalDeviceFeatures(),
    queue_config = dictionary([
        Vk.QUEUE_GRAPHICS_BIT | Vk.QUEUE_COMPUTE_BIT => 1
    ]),
    with_validation = true,
    debug = true,
)

    if with_validation && "VK_LAYER_KHRONOS_validation" ∉ instance_layers
        push!(instance_layers, "VK_LAYER_KHRONOS_validation")
    end
    if debug && "VK_EXT_debug_utils" ∉ instance_extensions
        push!(instance_extensions, "VK_EXT_debug_utils")
    end

    available_layers = unwrap(Vk.enumerate_instance_layer_properties())
    unsupported_layers = filter(!in(getproperty.(available_layers, :layer_name)), instance_layers)
    if !isempty(unsupported_layers)
        error("Requesting unsupported instance layers: $unsupported_layers")
    end

    available_extensions = unwrap(Vk.enumerate_instance_extension_properties())
    unsupported_extensions = filter(!in(getproperty.(available_extensions, :extension_name)), instance_extensions)
    if !isempty(unsupported_extensions)
        error("Requesting unsupported instance extensions: $unsupported_extensions")
    end

    dbg_info = if debug
        Vk.DebugUtilsMessengerCreateInfoEXT(
            |(
                Vk.DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT,
                Vk.DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT,
                Vk.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
                Vk.DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            ),
            |(
                Vk.DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT,
                Vk.DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
                Vk.DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
            ),
            debug_callback_c[],
        )
    else
        nothing
    end

    instance = Instance(instance_layers, instance_extensions, dbg_info; application_info)

    physical_device = first(unwrap(Vk.enumerate_physical_devices(instance)))

    # TODO: check for supported device features
    available_extensions = unwrap(Vk.enumerate_device_extension_properties(physical_device))
    unsupported_extensions = filter(!in(getproperty.(available_extensions, :extension_name)), device_extensions)
    if !isempty(unsupported_extensions)
        error("Requesting unsupported device extensions: $unsupported_extensions")
    end

    device = Device(physical_device, device_extensions, queue_config; enabled_features)

    instance, device
end

function debug_messenger(instance, info::Vk.DebugUtilsMessengerCreateInfoEXT)
    unwrap(create(Vk.DebugUtilsMessengerEXT, instance, info))
end
