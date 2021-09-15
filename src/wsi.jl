struct Surface{T} <: LavaAbstraction
    handle::Vk.SurfaceKHR
    target::T
end

vk_handle_type(::Type{Surface{T}}) where {T} = (Vk.SurfaceKHR,T)

function Surface(win::XCBWindow)
    handle = create(Surface{XCBWindow}, Vk.XcbSurfaceCreateInfo(win.conn.h, win.id))
    Surface(handle, win)
end

struct Swapchain{T}
    handle::Vk.SwapchainKHR
    surface::Surface{T}
end

vk_handle_type(::Type{<:Swapchain}) = Vk.SwapchainKHR
