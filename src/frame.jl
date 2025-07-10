mutable struct Frame
    # Swapchain image.
    const image::Image
    # Semaphore signaled by the driver when the swapchain image
    # that has been acquired is now ready for use.
    const image_acquired::BinarySemaphore
    # This semaphore is signaled by the rendering queue to ensure presentation
    # happens after the image has been fully written to.
    const may_present::BinarySemaphore
    # This field is `nothing` until the first render has been submitted.
    # This fence is signaled by the rendering queue when all rendering work has finished.
    has_rendered::Optional{Fence}
end

function Frame(device::Device, image::Image, index, frame_count)
    image_acquired = BinarySemaphore(device)
    may_present = BinarySemaphore(device)
    Vk.set_debug_name(image_acquired, Symbol(:image_acquired_, index, :_, frame_count))
    Vk.set_debug_name(may_present, Symbol(:may_present_, index, :_, frame_count))
    Frame(image, image_acquired, may_present, nothing)
end

function free!(frame::Frame)
    @debug "Finalizing frame $frame"
    finalize(frame.image)
    finalize(frame.image_acquired)
    finalize(frame.may_present)
end

mutable struct FrameCycle{T}
    const device::Device
    swapchain::Swapchain{T}
    frames::Vector{Frame}
    frame_index::Int64
    frame_count::Int64
    next_image::Int64
    outdated::Bool
    pending_frames::Dictionary{Swapchain, Dictionary{Frame, Vector{Fence}}}
end

function FrameCycle(device::Device, swapchain::Swapchain)
    frames = get_frames_from_swapchain(device, swapchain, 0)
    FrameCycle(device, swapchain, frames, lastindex(frames), 0, -1, true, Dictionary{Swapchain, Dictionary{Frame, Vector{Fence}}}())
end

function FrameCycle(device::Device, surface::Surface; swapchain_kwargs...)
    FrameCycle(device, Swapchain(device, surface, Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_TRANSFER_DST_BIT; swapchain_kwargs...))
end

function surface_capabilities(fc::FrameCycle)
    info = Vk.PhysicalDeviceSurfaceInfo2KHR(; fc.swapchain.surface)
    capabilities = unwrap(Vk.get_physical_device_surface_capabilities_2_khr(fc.device.handle.physical_device, info))
    capabilities.surface_capabilities
end

current_frame(fc::FrameCycle) = fc.frames[fc.frame_index]

Base.collect(::Type{T}, fc::FrameCycle) where {T} = collect(T, current_frame(fc).image, fc.device)
Base.collect(fc::FrameCycle) = collect(current_frame(fc).image, fc.device)

function recreate_swapchain!(fc::FrameCycle, capabilities::Vk.SurfaceCapabilitiesKHR)
    (; swapchain) = fc
    info = setproperties(swapchain.info; old_swapchain = swapchain.handle, image_extent = capabilities.current_extent)
    handle = unwrap(Vk.create_swapchain_khr(fc.device, info))
    depends_on(handle, swapchain.surface)
    fc.swapchain = setproperties(swapchain, (; info, handle))
end

function get_frames_from_swapchain(device, swapchain, frame_count)
    frames = Frame[]
    for (i, handle) in enumerate(unwrap(Vk.get_swapchain_images_khr(device, swapchain)))
        img = image_wsi(handle, swapchain.info)
        Vk.set_debug_name(img, "swapchain_image_$i")
        push!(frames, Frame(device, img, i, frame_count))
    end
    frames
end

function is_outdated(fc::FrameCycle)
    fc.outdated && return true
    extent_has_changed(fc)
end

function extent_has_changed(fc::FrameCycle, capabilities = surface_capabilities(fc))
    (; current_extent) = capabilities
    (; image_extent) = fc.swapchain.info
    current_extent ≠ image_extent
end

@enum FrameCycleStatus begin
    FRAME_CYCLE_FIRST_FRAME = 1
    FRAME_CYCLE_SWAPCHAIN_RECREATED = 2
    FRAME_CYCLE_RENDERING_FRAME = 3
end

function recreate!(fc::FrameCycle)
    @debug string("Recreating on frame ", fc.frame_count, " (last index: ", fc.frame_index, ", new index: ", lastindex(fc.frames), ')')
    capabilities = surface_capabilities(fc)
    extent_has_changed(fc, capabilities) && recreate_swapchain!(fc, capabilities)
    fc.frames = get_frames_from_swapchain(fc.device, fc.swapchain, fc.frame_count)
    fc.frame_index = lastindex(fc.frames)
    fc.next_image = -1
    fc.outdated = false
    fc
end

function recreate!(f::F, fc::FrameCycle) where {F}
    recreate!(fc)
    f(FRAME_CYCLE_SWAPCHAIN_RECREATED)
end

function next_frame!(fc::FrameCycle)
    fc.frame_index = fc.next_image
    fc.next_image = -1
    fc.frame_count += 1
    current_frame(fc)
end

"""
Acquire the next image.
"""
function acquire_next_image!(fc::FrameCycle)
    (; image_acquired) = current_frame(fc)
    # When this returns, we may have the index of the next swapchain,
    # but operations on the image may still be ongoing until the
    # `image_acquired` semaphore has been signaled.
    status = Vk.acquire_next_image_khr(fc.device, fc.swapchain, 0; semaphore = image_acquired)
    !iserror(status) && ((idx, result) = unwrap(status))
    if iserror(status)
        err = unwrap_error(status)
        err.code == Vk.ERROR_OUT_OF_DATE_KHR && return err.code
        error("Could not acquire the next swapchain image ($(err.code))")
    end

    in(result, (Vk.SUBOPTIMAL_KHR, Vk.SUCCESS)) && (fc.next_image = idx + 1)
    fc.outdated |= result == Vk.SUBOPTIMAL_KHR && extent_has_changed(fc)
    result
end

function free_pending_frames!(fc::FrameCycle)
    for (swapchain, history) in pairs(fc.pending_frames)
        swapchain_in_use = swapchain.handle === fc.swapchain.handle
        for (frame, fences) in pairs(history)
            filter!(fences) do fence
                wait(fence, 0) || return true
                recycle!(fence)
                false
            end
            swapchain_in_use |= !isempty(fences)
        end
        swapchain_in_use && continue
        for frame in keys(history)
            free!(frame)
        end
        finalize(swapchain.handle)
        delete!(fc.pending_frames, swapchain)
    end
end

function cycle!(f, fc::FrameCycle)
    is_outdated(fc) && recreate!(f, fc)

    if fc.next_image == -1
        @timeit to "Acquire next image" ret = acquire_next_image!(fc)
        ret === Vk.NOT_READY && return nothing
        while ret === Vk.ERROR_OUT_OF_DATE_KHR
            recreate!(f, fc)
            @timeit to "Acquire next image" ret = acquire_next_image!(fc)
            ret === Vk.NOT_READY && return nothing
            yield()
        end
    end
    next = fc.frames[fc.next_image]
    if !isnothing(next.has_rendered)
        wait(next.has_rendered, 0) || return nothing
        reset(next.has_rendered)
    end
    state = render_and_present!(f, fc)
    free_pending_frames!(fc)
    state
end

function render_and_present!(f, fc::FrameCycle)
    (; swapchain, device) = fc
    last_frame = current_frame(fc)
    (; image_acquired) = last_frame
    frame = next_frame!(fc)

    # Submit rendering commands.
    @timeit to "Submit rendering commands" begin
        # TODO: Pass in `image_acquired` to `f` such that typically only the last bit of the rendering
        # waits for the image. It is indeed most likely that the user will perform rendering operations
        # on another image, and only at the very end transfer the frame's contents to the swapchain image.
        fc.frame_count == 1 && f(FRAME_CYCLE_FIRST_FRAME)
        @timeit to "User-specified cycle function" submission = f(FRAME_CYCLE_RENDERING_FRAME)
        isnothing(submission) && return nothing
        isa(submission, SubmissionInfo) || throw(ArgumentError("A `SubmissionInfo` must be returned to properly synchronize with frame presentation."))
        push!(submission.wait_semaphores, Vk.SemaphoreSubmitInfo(image_acquired.handle, 0, 0; stage_mask = Vk.PIPELINE_STAGE_2_ALL_COMMANDS_BIT))
        push!(submission.signal_semaphores, Vk.SemaphoreSubmitInfo(frame.may_present.handle, 0, 0; stage_mask = Vk.PIPELINE_STAGE_2_ALL_COMMANDS_BIT))
        push!(submission.release_after_completion, frame)
        @timeit to "Submission" state = submit(device.queues, submission)
    end

    # Submit the presentation command.
    @timeit to "Submit presentation commands" begin
        # Fence to be signaled by the driver when the presentation queue
        # is done with swapchain-related resources.
        # These resources include the swapchain image, its bound memory, and
        # the wait semaphore used when queuing the image for presentation.
        has_presented = get_fence!(device)
        Vk.set_debug_name(has_presented, Symbol(:has_presented_, fc.frame_index, :_, fc.frame_count))
        present_fence_info = Vk.SwapchainPresentFenceInfoEXT([has_presented.handle])
        present_info = Vk.PresentInfoKHR([frame.may_present.handle], [swapchain.handle], [fc.frame_index - 1]; next = present_fence_info)
        record_pending!(fc, frame, has_presented)
        ret = Vk.queue_present_khr(swapchain.queue, present_info)
        if iserror(ret)
            (; code) = unwrap_error(ret)
            # Allow out of date errors, but throw if others are encountered.
            code ≠ Vk.ERROR_OUT_OF_DATE_KHR && unwrap(ret)
        else
            code = unwrap(ret)
        end
        code == Vk.ERROR_OUT_OF_DATE_KHR || code == Vk.SUBOPTIMAL_KHR && recreate!(f, fc)
    end

    state
end

function record_pending!(fc::FrameCycle, frame::Frame, fence::Fence)
    pending_frames = get!(Dictionary{Frame, Vector{Fence}}, fc.pending_frames, fc.swapchain)
    fences = get!(Vector{Fence}, pending_frames, frame)
    push!(fences, fence)
end

function initialize_for_presentation!(rg::RenderGraph, target::Resource, frame::Frame)
    image = Resource(frame.image)
    transfer = transfer_command(target, image)
    present = present_command(image)
    add_nodes!(rg, transfer, present)
end

function render!(rg::RenderGraph, frame::Frame)
    # XXX: reuse command buffer(s) from RenderGraph.
    command_buffer = request_command_buffer(rg.device)
    render!(rg, command_buffer)
    isa(frame.has_rendered, Fence) && recycle!(frame.has_rendered)
    has_rendered = get_fence!(rg.device)
    frame.has_rendered = has_rendered
    Vk.set_debug_name(has_rendered, :has_rendered)
    SubmissionInfo(command_buffers = [Vk.CommandBufferSubmitInfo(command_buffer)], release_after_completion = [rg], queue_family = command_buffer.queue_family_index, signal_fence = has_rendered)
end

function draw_and_prepare_for_presentation(device::Device, nodes, target::Resource, frame::Frame)
    rg = RenderGraph(device, nodes)
    initialize_for_presentation!(rg, target, frame)
    render!(rg, frame)
end
