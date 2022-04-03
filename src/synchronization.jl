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
  fence::Optional{Vk.Fence}
  free_after_completion::Vector{Any}
  release_after_completion::Vector{Any}
end

function ExecutionState(queue; fence = nothing, free_after_completion = [], release_after_completion = [])
  ExecutionState(queue, fence, free_after_completion, release_after_completion)
end

function finalize!(exec::ExecutionState)
  for resource in exec.free_after_completion
    finalize(resource)
  end
  empty!(exec.free_after_completion)
  empty!(exec.release_after_completion)
  true
end

function _wait(fence::Vk.Fence, timeout)
  ret = unwrap(Vk.wait_for_fences(Vk.device(fence), [fence], true, timeout))
  ret == Vk.SUCCESS
end

function _wait(fences, timeout)
  isempty(fences) && return true
  ret = unwrap(Vk.wait_for_fences(Vk.device(first(fence)), fences, true, timeout))
  ret == Vk.SUCCESS
end

function Base.wait(exec::ExecutionState, timeout = typemax(UInt32))
  fence = exec.fence::Vk.Fence
  _wait(fence, timeout) && finalize!(exec)
end

function Base.wait(execs::AbstractVector{ExecutionState}, timeout = typemax(UInt32))
  fences = map(Base.Fix2(getproperty, :fence), execs)::Vector{Vk.Fence}
  _wait(fences, timeout) && all(finalize!, execs)
end
Base.wait((x, exec)::Tuple{<:Any,ExecutionState}) = wait(exec) && return x

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
  ExecutionState(q; fence = info.signal_fence, info.free_after_completion, info.release_after_completion)
end

function Base.show(io::IO, exec::ExecutionState)
  print(io, ExecutionState, "($(exec.queue)")
  if !isnothing(exec.fence)
    is_complete = _wait(exec.fence, 0)
    if is_complete
      print(io, ", completed execution")
    else
      print(io, ", in progress")
    end
  end
  print(io, ')')
end

struct FencePool
  device::Vk.Device
  available::Vector{Vk.Fence}
  pending::Vector{Vk.Fence}
end

FencePool(device) = FencePool(device, [Vk.Fence(device) for _ in 1:10], [])

function compact!(pool::FencePool)
  to_reset = Vk.Fence[]
  filter!(pool.pending) do fence
    res = unwrap(Vk.get_fence_status(pool.device, fence))
    res == Vk.NOT_READY && return true
    push!(to_reset, fence)
    push!(pool.available, fence)
    false
  end
  !isempty(to_reset) && unwrap(Vk.reset_fences(pool.device, to_reset))
  nothing
end

function fence(pool::FencePool)
  isempty(pool.available) && compact!(pool)
  fence = isempty(pool.available) ? Vk.Fence(pool.device) : pop!(pool.available)
  push!(pool.pending, fence)
  fence
end
