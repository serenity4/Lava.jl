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

const REQUIRED_DEVICE_EXTENSIONS = [
    "VK_KHR_synchronization2",
]

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
    message_types = Vk.DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT | Vk.DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
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
            message_types,
            debug_callback_c[],
        )
    else
        nothing
    end

    instance = Instance(instance_layers, instance_extensions, dbg_info; application_info)

    physical_device = first(unwrap(Vk.enumerate_physical_devices(instance)))

    # TODO: check for supported device features
    available_extensions = unwrap(Vk.enumerate_device_extension_properties(physical_device))
    union!(device_extensions, REQUIRED_DEVICE_EXTENSIONS)
    unsupported_extensions = filter(!in(getproperty.(available_extensions, :extension_name)), device_extensions)
    if !isempty(unsupported_extensions)
        error("Requesting unsupported device extensions: $unsupported_extensions")
    end

    descriptor_indexing = descriptor_indexing_features()

    device = Device(physical_device, device_extensions, queue_config; enabled_features, next = descriptor_indexing)

    instance, device
end

function descriptor_indexing_features(features::Symbol...)
    T = Vk.PhysicalDeviceDescriptorIndexingFeatures
    fields = map(in(features), filter(≠(:next), fieldnames(T)))
    T(fields...)
end

function debug_messenger(instance, info::Vk.DebugUtilsMessengerCreateInfoEXT)
    unwrap(create(Vk.DebugUtilsMessengerEXT, instance, info))
end
