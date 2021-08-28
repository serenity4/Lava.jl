import Vulkan

import Vulkan: handle

const Vk = Vulkan
using .Vk.ResultTypes: unwrap

abstract type CreateInfo end

abstract type Handle end
abstract type Instance <: Handle end
abstract type Device <: Handle end
abstract type Fence <: Handle end
abstract type Event <: Handle end
abstract type Semaphore <: Handle end

(H::Type{<:Handle})(args...; kwargs...) = unwrap(create(H, args...; kwargs...))

create(::Type{Instance}, create_info; kwargs...) = Vk.create_instance(convert(Vk.InstanceCreateInfo, create_info); kwargs...)
create(::Type{Device}, physical_device, create_info; kwargs...) = Vk.create_device(physical_device, convert(Vk.DeviceCreateInfo, create_info); kwargs...)
create(::Type{Fence}, device, create_info; kwargs...) = Vk.create_fence(device, convert(Vk.FenceCreateInfo, create_info); kwargs...)
create(::Type{Event}, device; kwargs...) = Vk.create_event(device; kwargs...)
create(::Type{Semaphore}, device, create_info; kwargs...) = Vk.create_semaphore(device, convert(Vk.SemaphoreCreateInfo, create_info); kwargs...)

struct Created{H,I}
    handle::H
    info::I
    Created{H,I}(handle::H, info::I) where {H,I} = new{H,I}(handle, info)
end

Base.broadcastable(c::Created) = Ref(c)

handle(created::Created) = created.handle
info(created::Created) = created.info

for H in (:Instance, :Device, :Fence, :Semaphore)
    CI = Symbol(H, :CreateInfo)
    @eval CreateInfo(::Type{<:$H}) = Vk.$CI
    @eval Base.convert(T::Type{Vk.$CI}, args::Tuple) = T(args...)
    @eval Base.convert(T::Type{Vk.$CI}, created::Created{Vk.$H}) = info(created)
end

Base.convert(::Type{H}, created::Created{H}) where {H<:Vk.Handle} = handle(created)
Base.cconvert(T::Type{Ptr{Nothing}}, created::Created) = Base.cconvert(T, handle(created))

Created(::Type{T}, args...; kwargs...) where {T} = Created{T}(args...; kwargs...)

create(::Type{<:Created{H}}, args...; kwargs...) where {H} = create(H, args...; kwargs...)

function (C::Type{<:Created{H}})(info; kwargs...) where {H}
    CI = CreateInfo(H)
    info = convert(CI, info)
    handle = unwrap(create(C, info; kwargs...))
    Created{typeof(handle),CI}(handle, info)
end

function (C::Type{<:Created{H}})(parent_handle, info; kwargs...) where {H}
    CI = CreateInfo(H)
    info = convert(CI, info)
    handle = unwrap(create(C, parent_handle, info; kwargs...))
    Created{typeof(handle),CI}(handle, info)
end

instance = Instance(Vk.InstanceCreateInfo([], []))
physical_device = first(unwrap(Vk.enumerate_physical_devices(instance)))
device = Device(physical_device, Vk.DeviceCreateInfo([], [], []))

instance = Created{Instance}(Vk.InstanceCreateInfo([], []))
physical_device = first(unwrap(Vk.enumerate_physical_devices(instance)))
device = Created(Device, physical_device, ([Vk.DeviceQueueCreateInfo(0, [1.0])], [], []))
fence = Created(Fence, device, ())
fence2 = Created(Fence, device, fence)

sem = Semaphore(device, ())
sem2 = Created(Semaphore, device, ())
event = Event(device)
