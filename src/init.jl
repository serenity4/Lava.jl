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
    device_specific_features::AbstractVector{Symbol} = Symbol[],
    device_vulkan_features::AbstractVector{Symbol} = Symbol[],
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

    union!(device_vulkan_features, [:buffer_device_address])
    vulkan_features = physical_device_features(Vk.PhysicalDeviceVulkan12Features, device_vulkan_features)
    device_features = physical_device_features(Vk.PhysicalDeviceFeatures, device_specific_features)
    enabled_features = Vk.PhysicalDeviceFeatures2(device_features; next = vulkan_features)

    physical_device = pick_supported_device(unwrap(Vk.enumerate_physical_devices(instance)), enabled_features)

    available_extensions = unwrap(Vk.enumerate_device_extension_properties(physical_device))
    union!(device_extensions, REQUIRED_DEVICE_EXTENSIONS)
    unsupported_extensions = filter(!in(getproperty.(available_extensions, :extension_name)), device_extensions)
    if !isempty(unsupported_extensions)
        error("Requesting unsupported device extensions: $unsupported_extensions")
    end

    device = Device(physical_device, device_extensions, queue_config, enabled_features; next = enabled_features)

    instance, device
end

function physical_device_features(@nospecialize(T), features; next = C_NULL)
    names = fieldnames(T)
    unknown = filter(!in(names), features)
    if !isempty(unknown)
        error("Trying to set unknown features: $unknown")
    end
    fields = map(in(features), filter(≠(:next), names))
    if :next in names
        T(fields...; next)
    else
        T(fields...)
    end
end

function pick_supported_device(physical_devices, features)
    unsupported = nothing
    for pdevice in physical_devices
        # TODO: fix Vk.get_physical_device_features_2 in Vulkan.jl

        # initialize structure
        original = Vk._PhysicalDeviceFeatures2(
            Vk._PhysicalDeviceFeatures(ntuple(Returns(false), fieldcount(Vk.PhysicalDeviceFeatures))...);
            next = Vk._PhysicalDeviceVulkan12Features(ntuple(Returns(false), fieldcount(Vk.PhysicalDeviceVulkan12Features) - 1)...)
        )

        # build reference
        original_vk = Base.unsafe_convert(Vk.core.VkPhysicalDeviceFeatures2, original)
        ref = Ref(original_vk)

        # fill it with data
        GC.@preserve original Vk.core.vkGetPhysicalDeviceFeatures2(pdevice, ref)
        filled = ref[]

        # load next chain
        filled_next = unsafe_load(convert(Ptr{Vk.core.VkPhysicalDeviceVulkan12Features}, filled.pNext))

        # reconstruct wrapper structs (equivalent to `Vk.from_vk` that is defined for other structs, but by hand)
        vulkan_features = Vk.PhysicalDeviceVulkan12Features((getproperty(filled_next, name) for name in fieldnames(Vk.core.VkPhysicalDeviceVulkan12Features)[2:end])...)
        device_specific_features = Vk.PhysicalDeviceFeatures((getproperty(filled.features, name) for name in fieldnames(Vk.core.VkPhysicalDeviceFeatures))...)

        pdevice_features = Vk.PhysicalDeviceFeatures2(device_specific_features; next = vulkan_features)

        unsupported = unsupported_features(features, pdevice_features)
        isempty(unsupported) && return pdevice
    end
    throw("Physical device features $unsupported are required but not available on any device.")
end

function unsupported_features(requested::Vk.PhysicalDeviceFeatures2, available::Vk.PhysicalDeviceFeatures2)
    d = Dictionary{Symbol,Vector{Symbol}}()
    unsupported_vulkan = unsupported_features(requested.next, available.next)
    isempty(unsupported_vulkan) || insert!(d, :vulkan, unsupported_vulkan)
    unsupported_device = unsupported_features(requested.features, available.features)
    isempty(unsupported_device) || insert!(d, :device, unsupported_device)
    d
end

function unsupported_features(requested::T, available::T) where {T}
    filter(collect(fieldnames(T))) do name
        name == :next && return false
        getproperty(requested, name) && !getproperty(available, name)
    end
end

function debug_messenger(instance, info::Vk.DebugUtilsMessengerCreateInfoEXT)
    unwrap(create(Vk.DebugUtilsMessengerEXT, instance, info))
end
