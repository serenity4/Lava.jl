struct Instance <: LavaAbstraction
  handle::Vk.Instance
  layers::Vector{String}
  extensions::Vector{String}
  messenger::Optional{Vk.DebugUtilsMessengerEXT}
end

vk_handle_type(::Type{Instance}) = Vk.Instance

Instance(handle::Vk.Instance) = Instance(handle, [], [], nothing)

function Instance(layers, extensions, messenger_info::Optional{Vk.DebugUtilsMessengerCreateInfoEXT} = nothing; next = C_NULL, application_info = C_NULL)
  !isnothing(messenger_info) && (next = @set messenger_info.next = next)
  handle = unwrap(create(Instance, Vk.InstanceCreateInfo(layers, extensions; application_info, next)))
  if !isnothing(messenger_info)
    messenger = debug_messenger(handle, messenger_info)
  else
    messenger = nothing
  end
  Instance(handle, layers, extensions, messenger)
end

const REQUIRED_DEVICE_EXTENSIONS = String[
]

function init(;
  instance_layers = String[],
  instance_extensions = String[],
  vulkan_version = v"1.3.207", # always make sure to match the version of the Vulkan API wrapped by Vulkan.jl
  application_info = Vk.ApplicationInfo(v"1", v"1", vulkan_version),
  device_extensions = String[],
  device_specific_features::AbstractVector{Symbol} = Symbol[],
  device_vulkan_features::AbstractVector{Symbol} = Symbol[],
  queue_config = dictionary([
    Vk.QUEUE_GRAPHICS_BIT | Vk.QUEUE_COMPUTE_BIT => 1,
    Vk.QUEUE_TRANSFER_BIT => 1,
  ]),
  with_validation = true,
  debug = true,
  message_types = Vk.DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT | Vk.DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | Vk.DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT,
)

  vulkan_version ≥ v"1.3" || error("Lava requires Vulkan 1.3 or later.")
  if with_validation && "VK_LAYER_KHRONOS_validation" ∉ instance_layers
    push!(instance_layers, "VK_LAYER_KHRONOS_validation")
  end
  debug && union!(instance_extensions, ["VK_EXT_debug_utils"])
  union!(instance_extensions, ["VK_KHR_surface", "VK_KHR_get_surface_capabilities2"])

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

  next = C_NULL
  with_validation && (next = Vk.ValidationFeaturesEXT([Vk.VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT], []; next))

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
  end

  instance = Instance(instance_layers, instance_extensions, dbg_info; application_info, next)

  union!(
    device_vulkan_features,
    [:buffer_device_address, :descriptor_indexing, :descriptor_binding_partially_bound, :vulkan_memory_model, :synchronization2, :dynamic_rendering, :timeline_semaphore],
  )
  vulkan_features = physical_device_features_core(device_vulkan_features)
  union!(device_specific_features, [:shader_float_64, :shader_int_64, :sampler_anisotropy])
  device_features = physical_device_features(Vk.PhysicalDeviceFeatures, device_specific_features)
  enabled_features = Vk.PhysicalDeviceFeatures2(device_features; next = vulkan_features)

  physical_device = pick_supported_device(unwrap(Vk.enumerate_physical_devices(instance)), enabled_features)

  available_extensions = unwrap(Vk.enumerate_device_extension_properties(physical_device))
  union!(device_extensions, REQUIRED_DEVICE_EXTENSIONS)
  union!(device_extensions, ["VK_KHR_swapchain"])
  unsupported_extensions = filter(!in(getproperty.(available_extensions, :extension_name)), device_extensions)
  if !isempty(unsupported_extensions)
    error("Requesting unsupported device extensions: $unsupported_extensions")
  end

  device = Device(physical_device, application_info.api_version, device_extensions, queue_config, enabled_features; next = enabled_features)

  instance, device
end

function physical_device_features(@nospecialize(T), features)
  names = fieldnames(T)
  unknown = filter(!in(names), features)
  if !isempty(unknown)
    error("Trying to set unknown features: $unknown")
  end
  fields = map(in(features), filter(≠(:next), names))
  T(fields...)
end

function physical_device_features_core(features)
  names = Symbol[]
  f = C_NULL
  for T in (Vk.PhysicalDeviceVulkan11Features, Vk.PhysicalDeviceVulkan12Features, Vk.PhysicalDeviceVulkan13Features)
    version_features = fieldnames(T) ∩ features
    append!(names, version_features)
    new_f = physical_device_features(T, version_features)
    @reset new_f.next = f
    f = new_f
  end
  Set(names) == Set(features) || error("Trying to set unknown features: $(setdiff(features, names))")
  f
end

function pick_supported_device(physical_devices, features)
  unsupported = nothing
  for pdevice in physical_devices
    pdevice_features = Vk.get_physical_device_features_2(
      pdevice,
      # Vk.PhysicalDeviceVulkan13Features,
      # Vk.PhysicalDeviceVulkan12Features,
      # Vk.PhysicalDeviceVulkan11Features,
    )
    unsupported = unsupported_features(features, pdevice_features)
    isempty(unsupported) && return pdevice
  end
  throw("Physical device features $features are required but not available on any device.")
end

function unsupported_features(requested::Vk.PhysicalDeviceFeatures2, available::Vk.PhysicalDeviceFeatures2)
  d = Dictionary{Symbol,Vector{Symbol}}()
  # TODO: Reenable this once we figure out why the `next` chain of core Vulkan features produces segfaults.
  # unsupported_vulkan = unsupported_features(requested.next, available.next)
  # isempty(unsupported_vulkan) || insert!(d, :vulkan, unsupported_vulkan)
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
