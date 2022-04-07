struct Surface{T} <: LavaAbstraction
  handle::Vk.SurfaceKHR
  target::T
end

vk_handle_type(::Type{<:Surface}) = Vk.SurfaceKHR

struct Swapchain{T}
  handle::Vk.SwapchainKHR
  surface::Surface{T}
end

vk_handle_type(::Type{<:Swapchain}) = Vk.SwapchainKHR

"""
Opaque image that comes from the Window System Integration (WSI) as returned by `Vk.get_swapchain_images_khr`.
"""
struct ImageWSI <: Image{2,OpaqueMemory}
  handle::Vk.Image
end
