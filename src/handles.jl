Base.convert(T::Type{<:Vk.Handle}, x::LavaAbstraction) = handle(x)
Base.unsafe_convert(T::Type{Ptr{Cvoid}}, x::LavaAbstraction) = Base.unsafe_convert(T, handle(x))

Base.broadcastable(x::LavaAbstraction) = Ref(x)

"""
Opaque handle to a foreign API data structure. Necessary to interact with external libraries such as Vulkan.
"""
function handle end

handle(x) = Vk.handle(x)
handle(x::LavaAbstraction) = handle(x.handle)
Vk.handle(x::LavaAbstraction) = handle(x)

Base.finalize(x::LavaAbstraction) = finalize(handle(x))

Vk.device(x::LavaAbstraction) = Vk.device(handle(x))
device(x::LavaAbstraction) = Vk.device(handle(x))
device(x::Vk.Device) = x

physical_device(x::LavaAbstraction) = Vk.device(x).physical_device
physical_device(x::Vk.PhysicalDevice) = x
physical_device(device::Vk.Device) = device.physical_device

"""
    vk_handle_type(T)

Trait function for a type that wraps a specific Vulkan handle.
"""
function vk_handle_type end

create(::Type{T}, args...; kwargs...) where {T} = create(vk_handle_type(T), args...; kwargs...)
create(t::T, args...; kwargs...) where {T} = create(vk_handle_type(T), args...; kwargs...)

create(::Type{Vk.Instance}, create_info; kwargs...) = Vk.create_instance(convert(Vk.InstanceCreateInfo, create_info); kwargs...)
create(::Type{Vk.Device}, physical_device, create_info; kwargs...) =
  Vk.create_device(physical_device, convert(Vk.DeviceCreateInfo, create_info); kwargs...)
create(::Type{Vk.Fence}, device, create_info; kwargs...) = Vk.create_fence(device, convert(Vk.FenceCreateInfo, create_info); kwargs...)
create(::Type{Vk.Event}, device; kwargs...) = Vk.create_event(device; kwargs...)
create(::Type{Vk.DebugUtilsMessengerEXT}, instance, create_info; kwargs...) =
  Vk.create_debug_utils_messenger_ext(instance, convert(Vk.DebugUtilsMessengerCreateInfoEXT, create_info); kwargs...)
create(::Type{Vk.DeviceMemory}, device, create_info; kwargs...) = Vk.allocate_memory(device, convert(Vk.MemoryAllocateInfo, create_info); kwargs...)
create(::Type{Vk.Buffer}, device, create_info; kwargs...) = Vk.create_buffer(device, convert(Vk.BufferCreateInfo, create_info); kwargs...)
create(::Type{Vk.Image}, device, create_info; kwargs...) = Vk.create_image(device, convert(Vk.ImageCreateInfo, create_info); kwargs...)
create(::Type{Vk.ImageView}, device, create_info; kwargs...) = Vk.create_image_view(device, convert(Vk.ImageViewCreateInfo, create_info); kwargs...)
create(::Tuple{Type{Vk.SurfaceKHR},Type{XCBWindow}}, instance, create_info; kwargs...) =
  Vk.create_xcb_surface_khr(instance, convert(Vk.XcbSurfaceCreateInfo, create_info); kwargs...)
create(::Type{Vk.DescriptorSetLayout}, device, bindings; kwargs...) =
  Vk.create_descriptor_set_layout(device, convert(Vk.DescriptorSetLayoutCreateInfo, bindings); kwargs...)
create(::Type{Vk.DescriptorPool}, device, create_info; kwargs...) =
  Vk.create_descriptor_pool(device, convert(Vk.DescriptorPoolCreateInfo, create_info); kwargs...)


has_parent(::Type{Vk.Device}) = true
parent_handle_type(::Type{Vk.Device}) = Vk.PhysicalDevice

has_parent(::Type{Vk.PhysicalDevice}) = true
parent_handle_type(::Type{Vk.PhysicalDevice}) = Vk.Instance

has_parent(::Type{Vk.Instance}) = false

has_parent(::Type{Vk.ImageView}) = true
parent_handle_type(::Type{Vk.ImageView}) = Vk.Device

has_parent(::Type{Vk.Sampler}) = true
parent_handle_type(::Type{Vk.Sampler}) = Vk.Device

@generated function empty_handle(::Type{T}) where {T<:Vk.Handle}
  if has_parent(T)
    Expr(:new, T, C_NULL, empty_handle(parent_handle_type(T)), Vk.RefCounter(0), 0)
  else
    Expr(:new, T, C_NULL, Vk.RefCounter(0), 0)
  end
end

@generated empty_handle(::Type{T}) where {T<:LavaAbstraction} = empty_handle(vk_handle_type(T))
