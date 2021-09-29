struct ExecutionState
    queue::Queue
    semaphore::Optional{Vk.Semaphore}
    fence::Optional{Vk.Fence}
end

function Base.wait(state::ExecutionState, timeout = typemax(UInt32))
    fence = state.fence::Vk.Fence
    ret = unwrap(Vk.wait_for_fences(fence.device, [fence], true, timeout))
    ret == Vk.SUCCESS
end

function Base.wait(states::AbstractVector{ExecutionState}, timeout = typemax(UInt32); wait_all = true)
    fences = map(Base.Fix2(getproperty, :fence), states)::Vector{Fence}
    ret = unwrap(Vk.wait_for_fences(fence.device, fences, wait_all, timeout))
    ret == Vk.SUCCESS
end

function submit(dispatch::QueueDispatch, queue_family_index, submit_infos; signal_fence = false, semaphore = nothing)
    q = queue(dispatch, queue_family_index)

    state = if signal_fence
        ExecutionState(q, semaphore, Vk.Fence(Vk.device(q)))
    else
        ExecutionState(q, semaphore, nothing)
    end

    unwrap(Vk.queue_submit_2_khr(q, submit_infos; fence = something(state.fence, C_NULL)))
    state
end

function submit(dispatch::QueueDispatch, queue_family_index, submit_info::Vk.SubmitInfo2KHR; signal_fence = false, semaphore = nothing)
    if !isnothing(semaphore)
        semaphore in submit_info.signal_semaphore_infos || error("The provided semaphore was not included in the submission structure")
    end
    submit(dispatch, queue_family_index, [submit_info]; signal_fence, semaphore)
end
