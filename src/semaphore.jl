abstract type Semaphore <: LavaAbstraction end

vk_handle_type(::Type{<:Semaphore}) = Vk.Semaphore

mutable struct TimelineSemaphore <: Semaphore
  const handle::Vk.Semaphore
  signal_value::UInt64
end

TimelineSemaphore(device) = TimelineSemaphore(Vk.Semaphore(device, next = Vk.SemaphoreTypeCreateInfo(Vk.SEMAPHORE_TYPE_TIMELINE, 0)), 0)
next_value!(semaphore::TimelineSemaphore) = semaphore.signal_value += 1

function timeline_semaphores(infos::AbstractVector{Vk.SemaphoreSubmitInfo})
  semaphores = TimelineSemaphore[]
  for info in infos
    !iszero(info.value) && push!(semaphores, TimelineSemaphore(info.semaphore, info.value))
  end
  semaphores
end

function Base.wait(semaphores::AbstractVector{TimelineSemaphore}, timeout)
  isempty(semaphores) && return true
  ret = unwrap(Vk.wait_semaphores(first(semaphores).handle.device, semaphore_wait_info(semaphores), timeout))
  ret == Vk.SUCCESS
end

function semaphore_wait_info(semaphores::AbstractVector{TimelineSemaphore})
  info = Vk.SemaphoreWaitInfo(Vk.Semaphore[], UInt64[])
  for semaphore in semaphores
    push!(info.semaphores, semaphore.handle)
    push!(info.values, semaphore.signal_value)
  end
  info
end

struct BinarySemaphore <: Semaphore
  handle::Vk.Semaphore
  BinarySemaphore(device) = new(Vk.Semaphore(device))
end
