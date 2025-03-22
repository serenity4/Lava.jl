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
  done::RefValue{Bool}
  queue::Queue
  fences::Vector{Fence}
  semaphores::Vector{TimelineSemaphore}
  command_buffers::Vector{Vk.CommandBuffer}
  free_after_completion::Vector{Any}
  release_after_completion::Vector{Any}
end

function ExecutionState(queue::Queue; command_buffers = Vk.CommandBuffer[], fences = Fence[], semaphores = TimelineSemaphore[], free_after_completion = [], release_after_completion = [])
  ExecutionState(Ref(false), queue, fences, semaphores, command_buffers, free_after_completion, release_after_completion)
end

function finalize!(exec::ExecutionState)
  for resource in exec.free_after_completion
    finalize(resource)
  end
  if !isempty(exec.command_buffers)
    cb = first(exec.command_buffers)
    Vk.free_command_buffers(Vk.device(cb), cb.command_pool, exec.command_buffers)
  end
  empty!(exec.free_after_completion)
  empty!(exec.release_after_completion)
  for fence in exec.fences
    recycle!(fence)
  end
end

function Base.wait(exec::ExecutionState, timeout = typemax(UInt32); finalize = true)
  exec.done[] && return true
  (!wait(exec.fences, timeout) || !wait(exec.semaphores, timeout)) && return false
  finalize && finalize!(exec)
  exec.done[] = true
  true
end

function Base.wait(execs::AbstractVector{ExecutionState}, timeout = typemax(UInt32); finalize = true)
  !all(exec -> wait(exec, timeout), execs) && return false
  finalize && foreach(finalize!, execs)
  true
end

function Base.wait((x, exec)::Tuple{<:Any,ExecutionState}, args...; kwargs...)
  wait(exec, args...; kwargs...)
  x
end

isdone(exec) = exec.done[]

Base.@kwdef mutable struct SubmissionInfo
  const wait_semaphores::Vector{Vk.SemaphoreSubmitInfo} = []
  const command_buffers::Vector{Vk.CommandBufferSubmitInfo} = []
  const signal_semaphores::Vector{Vk.SemaphoreSubmitInfo} = []
  const signal_fence::Optional{Fence} = nothing
  const release_after_completion::Vector{Any} = []
  const free_after_completion::Vector{Any} = []
  queue_family::Int64 = -1
end

function set_queue_family!(info::SubmissionInfo, index)
  info.queue_family == -1 || info.queue_family == index || throw(ArgumentError("A different queue family index has already been set"))
  info.queue_family = index
end

function submit(dispatch::QueueDispatch, info::SubmissionInfo)
  q = queue(dispatch, info.queue_family)
  command_buffers = Vk.CommandBuffer[]

  for cb_info in info.command_buffers
    end_recording(cb_info.command_buffer)
    push!(command_buffers, cb_info.command_buffer)
  end

  submit_info = Vk._SubmitInfo2(info.wait_semaphores, info.command_buffers, info.signal_semaphores)

  fence = isnothing(info.signal_fence) ? C_NULL : info.signal_fence.handle
  unwrap(Vk._queue_submit_2(q, [submit_info]; fence))
  fences = Fence[]
  !isnothing(info.signal_fence) && push!(fences, info.signal_fence)
  ExecutionState(q; command_buffers, fences, semaphores = timeline_semaphores(info.signal_semaphores), info.free_after_completion, info.release_after_completion)
end

function Base.show(io::IO, exec::ExecutionState)
  print(io, ExecutionState, '(')
  if isdone(exec)
    print(io, "completed execution")
  else
    print(io, "may not have completed execution")
  end
  print(io, " on queue ", exec.queue, ')')
end

sync_submission(device) = SubmissionInfo(signal_fence = get_fence!(device))
