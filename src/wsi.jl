struct Surface{T} <: LavaAbstraction
  handle::Vk.SurfaceKHR
  target::T
end

vk_handle_type(::Type{<:Surface}) = Vk.SurfaceKHR

struct Swapchain{T} <: LavaAbstraction
  handle::Vk.SwapchainKHR
  info::Vk.SwapchainCreateInfoKHR
  surface::Surface{T}
end

vk_handle_type(::Type{<:Swapchain}) = Vk.SwapchainKHR

list_print(f, values) = string("\n• ", join(map(f, values), "\n• "), '\n')
list_print(values) = list_print(identity, values)

function Swapchain(device::Device, surface::Surface, usage::Vk.ImageUsageFlag; n = 2, present_mode = Vk.PRESENT_MODE_IMMEDIATE_KHR, format = Vk.FORMAT_B8G8R8A8_SRGB, color_space = Vk.COLOR_SPACE_SRGB_NONLINEAR_KHR, composite_alpha = Vk.COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
  (; physical_device) = device.handle
  surface_info = Vk.PhysicalDeviceSurfaceInfo2KHR(; surface)
  capabilities = unwrap(Vk.get_physical_device_surface_capabilities_2_khr(physical_device, surface_info)).surface_capabilities

  capabilities.min_image_count ≤ n ≤ capabilities.max_image_count || error("The provided surface requires $(capabilities.min_image_count) ≤ n ≤ $(capabilities.max_image_count) (got $n)")
  usage in capabilities.supported_usage_flags || error("The surface does not support swapchain images with usage $usage. Supported flags are $(capabilities.supported_usage_flags)")
  composite_alpha in capabilities.supported_composite_alpha || error("The surface does not support the provided composite alpha $composite_alpha. Supported flags are $(capabilities.supported_composite_alpha)")

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
    capabilities.current_extent,
    1,
    usage,
    Vk.SHARING_MODE_EXCLUSIVE,
    [],
    capabilities.current_transform,
    composite_alpha,
    present_mode,
    false,
  )
  handle = unwrap(Vk.create_swapchain_khr(device, info))
  Swapchain(handle, info, surface)
end

"""
Opaque image that comes from the Window System Integration (WSI) as returned by `Vk.get_swapchain_images_khr`.
"""
struct ImageWSI <: Image{2,OpaqueMemory}
  handle::Vk.Image
  dims::NTuple{2,Int64}
  format::Vk.Format
  layers::Int64
  usage::Vk.ImageUsageFlag
  queue_family_indices::Vector{Int8}
  sharing_mode::Vk.SharingMode
  layout::RefValue{Vk.ImageLayout}
end

samples(::ImageWSI) = 1
mip_levels(::ImageWSI) = 1
memory(::ImageWSI) = nothing

function ImageWSI(handle, info::Vk.SwapchainCreateInfoKHR; layout = Vk.IMAGE_LAYOUT_UNDEFINED)
  ImageWSI(handle, (info.image_extent.width, info.image_extent.height), info.image_format, info.image_array_layers, info.image_usage, info.queue_family_indices, info.image_sharing_mode, Ref(layout))
end

function Base.collect(@nospecialize(T), image::ImageWSI, device::Device)
  image = convert(ImageBlock, image, device)
  collect(T, image, device)
end

function Base.convert(::Type{ImageBlock}, image::ImageWSI, device::Device)
  ImageBlock(image.handle, dims(image), format(image), samples(image), mip_levels(image), layers(image), usage(image), image.queue_family_indices, image.sharing_mode, false, image.layout, Ref(OpaqueMemory()))
end
