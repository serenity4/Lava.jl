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

struct BinarySemaphore <: Semaphore
  handle::Vk.Semaphore
  BinarySemaphore(device) = new(Vk.Semaphore(device))
end

"""
Execution state that encapsulates synchronization primitives and resources
bound to a command submission on the GPU.

Resources bound to the related execution can be either freed or released (dereferenced)
once the execution completes.
The execution is assumed to be completed when the execution
state has been waited on (see [`wait`](@ref)), or when an inherited execution state has
been completed execution.
"""
struct ExecutionState
  queue::Queue
  fences::Vector{Vk.Fence}
  semaphores::Vector{TimelineSemaphore}
  free_after_completion::Vector{Any}
  release_after_completion::Vector{Any}
end

function ExecutionState(queue::Queue; fences = Vk.Fence[], semaphores = TimelineSemaphore[], free_after_completion = [], release_after_completion = [])
  ExecutionState(queue, fences, semaphores, free_after_completion, release_after_completion)
end

function finalize!(exec::ExecutionState)
  for resource in exec.free_after_completion
    finalize(resource)
  end
  empty!(exec.free_after_completion)
  empty!(exec.release_after_completion)
end

function Base.wait(fences::AbstractVector{Vk.Fence}, timeout)
  isempty(fences) && return true
  ret = unwrap(Vk.wait_for_fences(Vk.device(first(fences)), fences, true, timeout))
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

function Base.wait(semaphores::AbstractVector{TimelineSemaphore}, timeout)
  isempty(semaphores) && return true
  ret = unwrap(Vk.wait_semaphores(first(semaphores).handle.device, semaphore_wait_info(semaphores), timeout))
  ret == Vk.SUCCESS
end

function Base.wait(exec::ExecutionState, timeout = typemax(UInt32); finalize = true)
  (!wait(exec.fences, timeout) || !wait(exec.semaphores, timeout)) && return false
  finalize && finalize!(exec)
  true
end

function Base.wait(execs::AbstractVector{ExecutionState}, timeout = typemax(UInt32); finalize = true)
  !all(Base.Fix2(wait, timeout), execs) && return false
  finalize && foreach(finalize!, execs)
  true
end

function Base.wait((x, exec)::Tuple{<:Any,ExecutionState}, args...; kwargs...)
  wait(exec, args...; kwargs...)
  x
end

isdone(exec) = wait(exec, 0; finalize = false)

Base.@kwdef struct SubmissionInfo
  wait_semaphores::Vector{Vk.SemaphoreSubmitInfo} = []
  command_buffers::Vector{Vk.CommandBufferSubmitInfo} = []
  signal_semaphores::Vector{Vk.SemaphoreSubmitInfo} = []
  signal_fence::Optional{Vk.Fence} = nothing
  release_after_completion::Vector{Any} = []
  free_after_completion::Vector{Any} = []
end

function submit(dispatch::QueueDispatch, queue_family_index, info::SubmissionInfo)
  q = queue(dispatch, queue_family_index)

  for cb_info in info.command_buffers
    end_recording(cb_info.command_buffer)
  end

  submit_info = Vk.SubmitInfo2(info.wait_semaphores, info.command_buffers, info.signal_semaphores)

  unwrap(Vk.queue_submit_2(q, [submit_info]; fence = something(info.signal_fence, C_NULL)))
  fences = Vk.Fence[]
  !isnothing(info.signal_fence) && push!(fences, info.signal_fence)
  ExecutionState(q; fences, semaphores = timeline_semaphores(info.signal_semaphores), info.free_after_completion, info.release_after_completion)
end

function Base.show(io::IO, exec::ExecutionState)
  print(io, ExecutionState, '(')
  if isdone(exec)
    print(io, "completed execution")
  else
    print(io, "in progress")
  end
  print(io, " on queue ", exec.queue, ')')
end

struct FencePool
  device::Vk.Device
  available::Vector{Vk.Fence}
  completed::Vector{Vk.Fence}
  pending::Vector{Vk.Fence}
end

FencePool(device) = FencePool(device, [Vk.Fence(device) for _ in 1:10], [], [])

function recycle_completed!(pool::FencePool)
  (; available, completed) = pool
  n = length(completed)
  if !iszero(n)
    unwrap(Vk.reset_fences(pool.device, completed))
    append!(available, completed)
    empty!(completed)
  end
  n
end

function compact!(pool::FencePool)
  !iszero(recycle_completed!(pool)) && return
  filter!(pool.pending) do fence
    fence_status(fence) == Vk.NOT_READY && return true
    push!(pool.completed, fence)
    false
  end
  recycle_completed!(pool)
end

fence_status(fence::Vk.Fence) = unwrap(Vk.get_fence_status(fence.device, fence))

function fence(pool::FencePool)
  (; available) = pool
  isempty(available) && compact!(pool)
  fence = isempty(available) ? Vk.Fence(pool.device) : pop!(available)
  push!(pool.pending, fence)
  fence
end

function Base.empty!(pool::FencePool)
  empty!(pool.available)
  empty!(pool.pending)
  pool
end

Base.show(io::IO, pool::FencePool) = print(io, FencePool, "(", pool.device, ", ", length(pool.available), " available fences, ", length(pool.pending), " pending execution)")
