mutable struct Frame
    view::ImageView{ImageWSI}
    image_acquired::Vk.Semaphore
    const image_rendered::Vk.Semaphore
end

function Frame(view::ImageView{ImageWSI})
    (; device) = view.image.handle
    Frame(view, Vk.Semaphore(device), Vk.Semaphore(device))
end

mutable struct FrameCycle
    const device::Device
    swapchain::Swapchain
    const frames::Vector{Frame}
    frame_index::Int64
    frame_count::Int64
end

function FrameCycle(device, swapchain::Swapchain)
    FrameCycle(device, swapchain, get_frames(device, swapchain), 1, 0)
end

function Vk.SurfaceCapabilitiesKHR(fc::FrameCycle)
    unwrap(get_physical_device_surface_capabilities_khr(fc.device.physical_device, fc.swapchain.surface))
end

current_frame(fc::FrameCycle) = fc.frames[fc.frame_index]

function recreate_swapchain!(fc::FrameCycle, new_extent::NTuple{2,Int64})
    (; swapchain) = fc
    info = setproperties(swapchain.info, old_swapchain = handle(swapchain), image_extent = Vk.Extent2D(new_extent...))
    handle = unwrap(Vk.create_swapchain_khr(fc.device, info))
    fc.swapchain = setproperties(swapchain, (; info, handle))
end

function get_frames(device, swapchain)
    frames = Frame[]
    for handle in unwrap(Vk.get_swapchain_images_khr(device, swapchain))
        img = ImageWSI(handle)
        view = View(img; swapchain.info.format)
        push!(frames, Frame(view))
    end
    frames
end

function recreate!(fc::FrameCycle)
    (; current_extent) = Vk.SurfaceCapabilitiesKHR(fc)
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
    (; swapchain) = fc.swapchain
    
    # Acquire the next image.
    # We pass in a semaphore to signal because the implementation
    # may not be done reading from the image when this returns.
    # We use the semaphore from the last frame because we don't know
    # which index we'll get, but we'll exchange the semaphores once we know.
    last_frame = current_frame(fc)
    (; image_acquired) = last_frame
    idx = @timeit to "Acquire next image" acquire_next_image(fc.device, swapchain, image_acquired)
    if isnothing(idx)
        # Recreate swapchain and frames.
        recreate!(fc)
        return cycle!(fc, dispatch, submission)
    end
    frame = next_frame!(fc, idx)

    # Exchange semaphores.
    last_frame.image_acquired = frame.image_acquired
    frame.image_acquired = image_acquired

    (; dispatch) = fc.device

    # Submit rendering commands.
    @timeit to "Submit rendering commands" begin
        submission = f()
        push!(submission.wait_semaphores, Vk.SemaphoreSubmitInfoKHR(image_acquired, 0, 0; stage_mask = PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR))
        push!(submission.signal_semaphores, Vk.SemaphoreSubmitInfoKHR(frame.image_acquired, 0, 0; stage_mask = PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR))
        submit(dispatch, get_queue_family(dispatch, QUEUE_GRAPHICS_BIT), submission)
    end

    # Submit the presentation command.
    @timeit to "Submit presentation commands" begin
        present_info = Vk.PresentInfoKHR([frame.image_rendered], [swapchain], [idx])
        unwrap(present(dispatch, present_info))
    end
end

function acquire_next_image(device, swapchain, semaphore)
    while true
        status = acquire_next_image_khr(device, swapchain, 0; semaphore)
        if !iserror(status)
            idx, result = unwrap(status)
            result in (Vk.SUCCESS, Vk.SUBOPTIMAL_KHR) && return idx + 1
        else
            err = unwrap_error(status)
            err.code == ERROR_OUT_OF_DATE_KHR && return nothing
            error("Could not acquire the next swapchain image ($(err.code))")
        end
        yield()
    end
end
