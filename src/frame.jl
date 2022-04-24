mutable struct Frame
    const image::ImageWSI
    image_acquired::BinarySemaphore
    may_present::BinarySemaphore
    const image_rendered::TimelineSemaphore
end

function Frame(image::ImageWSI)
    (; device) = image.handle
    Frame(image, BinarySemaphore(device), BinarySemaphore(device), TimelineSemaphore(device))
end

mutable struct FrameCycle{T}
    const device::Device
    swapchain::Swapchain{T}
    const frames::Vector{Frame}
    frame_index::Int64
    frame_count::Int64
end

function FrameCycle(device::Device, swapchain::Swapchain)
    FrameCycle(device, swapchain, get_frames(device, swapchain), 1, 0)
end

function surface_capabilities(fc::FrameCycle)
    unwrap(Vk.get_physical_device_surface_capabilities_2_khr(fc.device.handle.physical_device, Vk.PhysicalDeviceSurfaceInfo2KHR(; fc.swapchain.surface))).surface_capabilities
end

current_frame(fc::FrameCycle) = fc.frames[fc.frame_index]

Base.collect(::Type{T}, fc::FrameCycle) where {T} = collect(T, current_frame(fc).image, fc.device)

function recreate_swapchain!(fc::FrameCycle, new_extent::Vk.Extent2D)
    (; swapchain) = fc
    info = setproperties(swapchain.info, old_swapchain = swapchain.handle, image_extent = new_extent)
    handle = unwrap(Vk.create_swapchain_khr(fc.device, info))
    fc.swapchain = setproperties(swapchain, (; info, handle))
end

function get_frames(device, swapchain)
    frames = Frame[]
    for handle in unwrap(Vk.get_swapchain_images_khr(device, swapchain))
        img = ImageWSI(handle, swapchain.info)
        push!(frames, Frame(img))
    end
    frames
end

function recreate!(fc::FrameCycle)
    (; current_extent) = surface_capabilities(fc)
    if current_extent ≠ fc.swapchain.info.image_extent
        recreate_swapchain!(fc, current_extent)
    end
    fc.frames .= get_frames(fc.device, fc.swapchain)
end

function next_frame!(fc::FrameCycle, idx)
    fc.frame_index = idx
    fc.frame_count += 1
    current_frame(fc)
end

function cycle!(f, fc::FrameCycle)
    # Acquire the next image.
    # We pass in a semaphore to signal because the implementation
    # may not be done reading from the image when this returns.
    # We use the semaphore from the last frame because we don't know
    # which index we'll get, but we'll exchange the semaphores once we know.
    last_frame = current_frame(fc)
    (; image_acquired) = last_frame
    idx = @timeit to "Acquire next image" acquire_next_image(fc.device, fc.swapchain, image_acquired)
    if isnothing(idx)
        # Recreate swapchain and frames.
        recreate!(fc)
        return cycle!(f, fc)
    end
    frame = next_frame!(fc, idx)

    # Exchange semaphores.
    last_frame.image_acquired = frame.image_acquired
    frame.image_acquired = image_acquired

    (; queues) = fc.device

    # Submit rendering commands.
    @timeit to "Submit rendering commands" begin
        submission = f(frame.image)
        push!(submission.wait_semaphores, Vk.SemaphoreSubmitInfo(image_acquired.handle, 0, 0; stage_mask = Vk.PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR))
        push!(submission.signal_semaphores, Vk.SemaphoreSubmitInfo(frame.image_rendered.handle, next_value!(frame.image_rendered), 0; stage_mask = Vk.PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR))
        # For syncing with the presentation engine only.
        push!(submission.signal_semaphores, Vk.SemaphoreSubmitInfo(frame.may_present.handle, 0, 0; stage_mask = Vk.PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR))
        push!(submission.release_after_completion, frame)
        state = submit(queues, get_queue_family(queues, Vk.QUEUE_GRAPHICS_BIT), submission)
    end

    # Submit the presentation command.
    @timeit to "Submit presentation commands" begin
        present_info = Vk.PresentInfoKHR([frame.may_present], [fc.swapchain], [idx - 1])
        present(queues, present_info)
    end

    state
end

function acquire_next_image(device, swapchain, semaphore)
    t0 = time()
    has_warned = false
    while true
        status = Vk.acquire_next_image_khr(device, swapchain, 0; semaphore)
        if !iserror(status)
            idx, result = unwrap(status)
            result in (Vk.SUCCESS, Vk.SUBOPTIMAL_KHR) && return idx + 1
            if time() > t0 + 1 && result == Vk.NOT_READY && !has_warned
                @warn "No swapchain image has been acquired for 1 second, returning with the status code `NOT_READY`."
                has_warned = true
            end
        else
            err = unwrap_error(status)
            err.code == Vk.ERROR_OUT_OF_DATE_KHR && return nothing
            error("Could not acquire the next swapchain image ($(err.code))")
        end
        yield()
    end
end
