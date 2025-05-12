struct Surface{T} <: LavaAbstraction
  handle::Vk.SurfaceKHR
  target::T
end

vk_handle_type(::Type{<:Surface}) = Vk.SurfaceKHR

Surface(instance::Instance, target) = Surface(Vk.SurfaceKHR(instance, target), target)

struct Swapchain{T} <: LavaAbstraction
  handle::Vk.SwapchainKHR
  info::Vk.SwapchainCreateInfoKHR
  surface::Surface{T}
  queue::Queue
end

vk_handle_type(::Type{<:Swapchain}) = Vk.SwapchainKHR

list_print(f, values) = string("\n• ", join(map(f, values), "\n• "), '\n')
list_print(values) = list_print(identity, values)

struct SwapchainScaling
  preferred_scalings::NTuple{3, Vk.PresentScalingFlagEXT}
  preferred_gravities::NTuple{2, NTuple{3, Vk.PresentGravityFlagEXT}}
end

default_preferred_scalings() = (Vk.PRESENT_SCALING_STRETCH_BIT_EXT, Vk.PRESENT_SCALING_ASPECT_RATIO_STRETCH_BIT_EXT, Vk.PRESENT_SCALING_ONE_TO_ONE_BIT_EXT)
default_preferred_gravities() = (Vk.PRESENT_GRAVITY_CENTERED_BIT_EXT, Vk.PRESENT_GRAVITY_MIN_BIT_EXT, Vk.PRESENT_GRAVITY_MAX_BIT_EXT)

function SwapchainScaling(; preferred_scalings = default_preferred_scalings(), preferred_gravities = default_preferred_gravities())
  isa(preferred_gravities, NTuple{2}) || (preferred_gravities = (preferred_gravities, preferred_gravities))
  SwapchainScaling(preferred_scalings, preferred_gravities)
end

function find_supported_scaling_parameters(info::Vk.SurfacePresentScalingCapabilitiesEXT, swapchain_scaling::SwapchainScaling)
  (; preferred_scalings, preferred_gravities) = swapchain_scaling
  i = findfirst(scaling -> in(scaling, info.supported_present_scaling), preferred_scalings)
  scaling = isnothing(i) ? Vk.PresentScalingFlagEXT() : preferred_scalings[i]
  i = findfirst(gravity -> in(gravity, info.supported_present_gravity_x), preferred_gravities[1])
  gravity_x = !isnothing(i) ? preferred_gravities[1][i] : iszero(scaling) ? Vk.PresentScalingFlagEXT() : nothing
  i = findfirst(gravity -> in(gravity, info.supported_present_gravity_y), preferred_gravities[2])
  gravity_y = !isnothing(i) ? preferred_gravities[2][i] : iszero(scaling) ? Vk.PresentScalingFlagEXT() : nothing
  iszero(gravity_x) && (gravity_y = gravity_x)
  iszero(gravity_y) && (gravity_x = gravity_y)

  scaling, gravity_x, gravity_y
end

function scaling_info(info::Vk.SurfacePresentScalingCapabilitiesEXT, swapchain_scaling::SwapchainScaling)
  scaling, gravity_x, gravity_y = find_supported_scaling_parameters(info, swapchain_scaling)
  iszero(scaling) && return C_NULL
  Vk.SwapchainPresentScalingCreateInfoEXT(; scaling_behavior = scaling, present_gravity_x = gravity_x, present_gravity_y = gravity_y)
end

function Swapchain(device::Device, surface::Surface, usage_flags::Vk.ImageUsageFlag;
                   n = 2,
                   present_mode = Vk.PRESENT_MODE_IMMEDIATE_KHR, # VK_PRESENT_MODE_MAILBOX_KHR?
                   format = Vk.FORMAT_B8G8R8A8_SRGB,
                   color_space = Vk.COLOR_SPACE_SRGB_NONLINEAR_KHR,
                   composite_alpha = Vk.COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                   scaling::Optional{SwapchainScaling} = SwapchainScaling(),
  )
  (; physical_device) = device.handle

  surface_info = Vk.PhysicalDeviceSurfaceInfo2KHR(; surface, next = Vk.SurfacePresentModeEXT(present_mode))
  info = unwrap(Vk.get_physical_device_surface_capabilities_khr(physical_device, surface))
  additional_capabilities = unwrap(Vk.get_physical_device_surface_capabilities_2_khr(physical_device, surface_info, Vk.SurfacePresentScalingCapabilitiesEXT))
  flags = Vk.SWAPCHAIN_CREATE_DEFERRED_MEMORY_ALLOCATION_BIT_EXT
  next = scaling_info(additional_capabilities.next::Vk.SurfacePresentScalingCapabilitiesEXT, scaling)

  info.min_image_count ≤ n ≤ info.max_image_count || error("The provided surface requires $(info.min_image_count) ≤ n ≤ $(info.max_image_count) (got $n)")
  usage_flags in info.supported_usage_flags || error("The surface does not support swapchain images with usage $usage_flags. Supported flags are $(info.supported_usage_flags)")
  composite_alpha in info.supported_composite_alpha || error("The surface does not support the provided composite alpha $composite_alpha. Supported flags are $(info.supported_composite_alpha)")
  (; current_extent, current_transform) = info

  formats = unwrap(Vk.get_physical_device_surface_formats_khr(physical_device; surface))
  any(formats) do fmt
    fmt.format == format && fmt.color_space == color_space
  end || error("Unsupported combination of format and color space values. Supported combinations are:", list_print(x -> string("Format ", x.format, " with color space ", x.color_space), values))

  present_modes = unwrap(Vk.get_physical_device_surface_present_modes_khr(physical_device; surface))
  any(==(present_mode), present_modes) || error("Unsupported presentation mode $present_mode. Supported presentation modes are:", list_print(present_modes))

  info = Vk.SwapchainCreateInfoKHR(
    surface.handle,
    n,
    format,
    color_space,
    current_extent,
    1,
    usage_flags,
    Vk.SHARING_MODE_EXCLUSIVE,
    [],
    current_transform,
    composite_alpha,
    present_mode,
    false;
    flags,
    next,
  )
  handle = unwrap(Vk.create_swapchain_khr(device, info))
  swapchain = Swapchain(handle, info, surface, find_presentation_queue(device.queues, [surface]))
  depends_on(handle, surface)
  swapchain
end

"""
Opaque image that comes from the Window System Integration (WSI) as returned by `Vk.get_swapchain_images_khr`.
"""
function image_wsi(handle, info::Vk.SwapchainCreateInfoKHR; layout = Vk.IMAGE_LAYOUT_UNDEFINED)
  Image(handle, [info.image_extent.width, info.image_extent.height], Vk.ImageCreateFlag(), info.image_format, 1, info.image_array_layers, 1, info.image_usage, info.queue_family_indices, info.image_sharing_mode, false, SubresourceMap(info.image_array_layers, 1, layout), RefValue{Memory}(), true)
end
