mutable struct Frame
    const image::Image
    image_acquired::BinarySemaphore
    may_present::BinarySemaphore
    const image_rendered::TimelineSemaphore
end

function Frame(image::Image)
    (; device) = image.handle
    image_acquired = BinarySemaphore(device)
    may_present = BinarySemaphore(device)
    image_rendered = TimelineSemaphore(device)
    Frame(image, image_acquired, may_present, image_rendered)
end

mutable struct FrameCycle{T}
    const device::Device
    swapchain::Swapchain{T}
    const frames::Vector{Frame}
    frame_index::Int64
    frame_count::Int64
end

function FrameCycle(device::Device, swapchain::Swapchain)
    frames = frames_from_swapchain(device, swapchain)
    FrameCycle(device, swapchain, frames, lastindex(frames), 0)
end

function FrameCycle(device::Device, surface::Surface; swapchain_kwargs...)
    FrameCycle(device, Swapchain(device, surface, Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_TRANSFER_DST_BIT; swapchain_kwargs...))
end

function surface_capabilities(fc::FrameCycle)
    unwrap(Vk.get_physical_device_surface_capabilities_2_khr(fc.device.handle.physical_device, Vk.PhysicalDeviceSurfaceInfo2KHR(; fc.swapchain.surface))).surface_capabilities
end

current_frame(fc::FrameCycle) = fc.frames[fc.frame_index]

Base.collect(::Type{T}, fc::FrameCycle) where {T} = collect(T, current_frame(fc).image, fc.device)
Base.collect(fc::FrameCycle) = collect(current_frame(fc).image, fc.device)

function recreate_swapchain!(fc::FrameCycle, new_extent::Vk.Extent2D)
    (; swapchain) = fc
    info = setproperties(swapchain.info, old_swapchain = swapchain.handle, image_extent = new_extent)
    handle = unwrap(Vk.create_swapchain_khr(fc.device, info))
    fc.swapchain = setproperties(swapchain, (; info, handle))
end

function frames_from_swapchain(device, swapchain)
    frames = Frame[]
    for (i, handle) in enumerate(unwrap(Vk.get_swapchain_images_khr(device, swapchain)))
        img = image_wsi(handle, swapchain.info)
        Vk.set_debug_name(img, "swapchain_image_$i")
        push!(frames, Frame(img))
    end
    frames
end

function recreate!(fc::FrameCycle)
    (; current_extent) = surface_capabilities(fc)
    if current_extent ≠ fc.swapchain.info.image_extent
        recreate_swapchain!(fc, current_extent)
    end
    # Eagerly finalize old frame semaphores, as swapchain recreation may
    # happen at a fast rate during window resize we don't want to hang onto them
    # for longer than necessary.
    for frame in fc.frames
        finalize(frame.may_present)
        finalize(frame.image_acquired)
    end
    fc.frames .= frames_from_swapchain(fc.device, fc.swapchain)
end

function next_frame!(fc::FrameCycle, idx)
    fc.frame_index = idx
    fc.frame_count += 1
    current_frame(fc)
end

"""
Acquire the next image.
"""
function acquire_next_image(fc::FrameCycle)
    (; image_acquired) = current_frame(fc)
    # We pass in a semaphore to signal because the implementation
    # may not be done reading from the image when this returns.
    status = Vk.acquire_next_image_khr(fc.device, fc.swapchain, 0; semaphore = image_acquired)
    if !iserror(status)
        idx, result = unwrap(status)
        result in (Vk.SUCCESS, Vk.SUBOPTIMAL_KHR) && return idx + 1
        result
    else
        err = unwrap_error(status)
        if err.code == Vk.ERROR_OUT_OF_DATE_KHR
            recreate!(fc)
            Vk.ERROR_OUT_OF_DATE_KHR
        else
            error("Could not acquire the next swapchain image ($(err.code))")
        end
    end
end

function cycle!(f, fc::FrameCycle)
    idx = 0
    t0 = time()
    has_warned = false

    @timeit to "Acquire next image" begin
        while true
            ret = acquire_next_image(fc)
            if ret isa Int
                idx = ret
                break
            end
            if time() > t0 + 1 && ret === Vk.NOT_READY && !has_warned
                @warn "No swapchain image has been acquired for 1 second, returning with the status code `NOT_READY`."
                has_warned = true
            end
            yield()
        end
    end

    cycle!(f, fc, idx)
end

function cycle!(f, fc::FrameCycle, idx::Integer)
    (; swapchain, device) = fc
    last_frame = current_frame(fc)
    (; image_acquired) = last_frame
    frame = next_frame!(fc, idx)

    # Submit rendering commands.
    @timeit to "Submit rendering commands" begin
        @timeit to "User-specified cycle function" submission = f(frame.image)
        isnothing(submission) && return nothing
        isa(submission, SubmissionInfo) || throw(ArgumentError("A `SubmissionInfo` must be returned to properly synchronize with frame presentation."))
        push!(submission.wait_semaphores, Vk.SemaphoreSubmitInfo(image_acquired.handle, 0, 0; stage_mask = Vk.PIPELINE_STAGE_2_ALL_COMMANDS_BIT))
        push!(submission.signal_semaphores, Vk.SemaphoreSubmitInfo(frame.may_present.handle, 0, 0; stage_mask = Vk.PIPELINE_STAGE_2_ALL_COMMANDS_BIT))
        push!(submission.release_after_completion, frame)
        @timeit to "Submission" state = submit(device.queues, submission)
    end

    # Submit the presentation command.
    @timeit to "Submit presentation commands" begin
        present_info = Vk.PresentInfoKHR([frame.may_present.handle], [swapchain.handle], [idx - 1])
        ret = Vk.queue_present_khr(swapchain.queue, present_info)
        # Ignore out of date errors, but throw if others are encountered.
        iserror(ret) && unwrap_error(ret).code ≠ Vk.ERROR_OUT_OF_DATE_KHR && unwrap(ret)
    end

    state
end

function draw_and_prepare_for_presentation(device::Device, nodes, source::Resource, target::Resource)
    transfer = transfer_command(source, target)
    present = present_command(target)
    rg = RenderGraph(device, nodes)
    add_nodes!(rg, transfer, present)
    command_buffer = request_command_buffer(device)
    baked = render!(rg, command_buffer)
    SubmissionInfo(command_buffers = [Vk.CommandBufferSubmitInfo(command_buffer)], release_after_completion = [baked], queue_family = command_buffer.queue_family_index, signal_fence = fence(device))
end

draw_and_prepare_for_presentation(device::Device, nodes, source::Resource, target::Image) = draw_and_prepare_for_presentation(device, nodes, source, Resource(target))
