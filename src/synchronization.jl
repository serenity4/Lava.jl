"""
Execution state that encapsulates synchronization primitives and resources
bound to a command submission on the GPU.

Resources bound to the related execution can be either freed or released (dereferenced)
once the execution completes.
The execution is assumed to be completed when the execution
state has been waited on (see [`wait`](@ref)), or when an inherited execution state has
been completed execution.

Execution state can be inherited across submissions, where resource dependencies
are transfered over to a new execution state bound to a command that **must** include
the inherited `ExecutionState`'s semaphore as a wait semaphore.
"""
struct ExecutionState
  queue::Queue
  semaphore::Optional{Vk.SemaphoreSubmitInfoKHR}
  fence::Optional{Vk.Fence}
  free_after_completion::Vector{Ref}
  release_after_completion::Vector{Ref}
end

function ExecutionState(queue; semaphore = nothing, fence = nothing, free_after_completion = [], release_after_completion = [])
  ExecutionState(queue, semaphore, fence, free_after_completion, release_after_completion)
end

function finalize!(exec::ExecutionState)
  for resource in exec.free_after_completion
    finalize(resource[])
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
  fences = map(Base.Fix2(getproperty, :fence), execs)::Vector{Fence}
  _wait(fences, timeout) && all(finalize!, execs)
end
Base.wait((x, exec)::Tuple{<:Any,ExecutionState}) = wait(exec) && return x

function submit(dispatch::QueueDispatch, queue_family_index, submit_infos;
  signal_fence = false,
  semaphore = nothing,
  free_after_completion = [],
  release_after_completion = [],
  inherit = nothing,
  check_inheritance = true,
)
  q = queue(dispatch, queue_family_index)

  fence = signal_fence ? Vk.Fence(Vk.device(q)) : nothing

  if inherit isa ExecutionState
    append!(free_after_completion, inherit.free_after_completion)
    append!(release_after_completion, inherit.release_after_completion)
    empty!(inherit.free_after_completion)
    empty!(inherit.release_after_completion)
    if check_inheritance && !any(Base.Fix1(in, inherit.semaphore), submit_infos.wait_semaphores)
      error("No wait semaphore has been registered that
          matches the inherited state.")
    end
  end

  for submit_info in submit_infos
    for command_buffer_info in submit_info.command_buffer_infos
      end_recording(command_buffer_info.command_buffer)
    end
  end

  unwrap(Vk.queue_submit_2_khr(q, submit_infos; fence = something(fence, C_NULL)))
  ExecutionState(q; semaphore, fence, free_after_completion, release_after_completion)
end

function submit(dispatch::QueueDispatch, queue_family_index, submit_info::Vk.SubmitInfo2KHR;
  signal_fence = false,
  semaphore = nothing,
  free_after_completion = [],
  release_after_completion = [],
  inherit = nothing,
  check_inheritance = true,
)
  if !isnothing(semaphore)
    semaphore in submit_info.signal_semaphore_infos || error("The provided semaphore was not included in the submission structure")
  end
  submit(
    dispatch,
    queue_family_index,
    [submit_info];
    signal_fence,
    semaphore,
    free_after_completion,
    release_after_completion,
    inherit,
    check_inheritance,
  )
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
