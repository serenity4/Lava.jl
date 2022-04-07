struct FrameSynchronization
    image_acquired::Vk.Semaphore
    image_rendered::Vk.Semaphore
    has_rendered::Vk.Fence
end

FrameSynchronization(device) = FrameSynchronization(Vk.Semaphore(device), Vk.Semaphore(device), Vk.Fence(device; flags = FENCE_CREATE_SIGNALED_BIT))

mutable struct FrameState
    const device::Device
    swapchain::Swapchain
    const frames::Vector{Frame}
    current_frame::Frame
    frame_count::Int64
    const syncs::Dictionary{Frame,FrameSynchronization}
end

device(fs::FrameState) = fs.device

function FrameState(device, swapchain::Swapchain)
    max_in_flight = info(swapchain).min_image_count
    fs = FrameState(device, Ref(swapchain), [], Ref{Frame}(), Ref(0), Dictionary())
    update!(fs)
end

function Vk.SurfaceCapabilitiesKHR(fs::FrameState)
    unwrap(get_physical_device_surface_capabilities_khr(device(fs).physical_device, info(fs.swapchain[]).surface))
end

function recreate_swapchain!(fs::FrameState, new_extent::NTuple{2,Int64})
    swapchain = fs.swapchain[]
    swapchain_info = setproperties(info(swapchain), old_swapchain = handle(swapchain), image_extent = Vk.Extent2D(new_extent...))
    swapchain_handle = unwrap(create_swapchain_khr(device(fs), swapchain_info))
    fs.swapchain[] = Created(swapchain_handle, swapchain_info)
    fs
end

function get_frames(fs::FrameState)
    (; swapchain) = fs
    extent = info(swapchain).image_extent
    map(unwrap(get_swapchain_images_khr(device(fs), swapchain))) do img
        view = View(
            device(fs),
            img,
            IMAGE_VIEW_TYPE_2D,
            info(swapchain).image_format,
            ComponentMapping(fill(COMPONENT_SWIZZLE_IDENTITY, 4)...),
            ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1),
        )

        fb = Framebuffer(device(fs), fs.render_pass, [view], extent.width, extent.height, 1)
        Frame(img, view, fb)
    end
end

function update!(fs::FrameState)
    empty!(fs.frames)
    empty!(fs.syncs)

    # TODO: make fields of returned-only structs high-level in Vulkan.jl
    _extent = SurfaceCapabilitiesKHR(fs).current_extent.vks
    new_extent = Extent2D(_extent.width, _extent.height)
    if new_extent ≠ info(fs.swapchain[]).image_extent
        recreate_swapchain!(fs, new_extent)
    end

    append!(fs.frames, get_frames(fs))
    fs.current_frame[] = first(fs.frames)

    foreach(fs.frames) do frame
        insert!(fs.syncs, frame, FrameSynchronization(device(fs.render_pass)))
    end
    fs
end

function next_frame!(fs::FrameState, dispatch::QueueDispatch, submission::SubmissionInfo = SubmissionInfo())
    (; swapchain) = fs.swapchain
    
    # Acquire the next image.
    # We pass in a semaphore to signal because the implementation
    # may not be done reading from the image when this returns.
    (; image_acquired) = fs.syncs[fs.current_frame]
    idx = @timeit to "Acquire next image" acquire_next_image(fs.device, swapchain, image_acquired)
    if isnothing(idx)
        # Recreate swapchain.
        update!(fs)
        return next_frame!(fs, dispatch, submission)
    end
    frame = fs.frames[idx + 1]
    sync = fs.syncs[frame]

    # Submit rendering commands.
    @timeit to "Submit rendering commands" begin
        push!(submission.wait_semaphores, Vk.SemaphoreSubmitInfoKHR(image_acquired, 0, 0; stage_mask = PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR))
        push!(submission.signal_semaphores, Vk.SemaphoreSubmitInfoKHR(sync.has_rendered, 0, 0; stage_mask = PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR))
        submit(dispatch, get_queue_family(dispatch, QUEUE_GRAPHICS_BIT), submission)
    end

    # Submit presentation command.
    @timeit to "Submit presentation commands" begin
        present_info = Vk.PresentInfoKHR([sync.has_rendered], [swapchain], [idx])
        unwrap(present(dispatch, present_info))
    end

    fs.frame_count += 1
    fs.current_frame = frame
end

function acquire_next_image(device, swapchain, semaphore)
    while true
        status = acquire_next_image_khr(device, swapchain, 0; semaphore)
        if !iserror(status)
            idx, result = unwrap(status)
            result in (Vk.SUCCESS, Vk.SUBOPTIMAL_KHR) && return idx
        else
            err = unwrap_error(status)
            err.code == ERROR_OUT_OF_DATE_KHR && return nothing
            error("Could not acquire the next swapchain image ($(err.code))")
        end
        yield()
    end
end

function wait_hasrendered(fs::FrameState)
    wait_for_fences(device(fs), getproperty.(fs.syncs, :has_rendered), true, typemax(UInt64))
end
