struct ImageSynchronizationRequirements
    layout::Vk.ImageLayout
end

abstract type ResourceInfo end

struct ImageResourceInfo <: ResourceInfo
    dims::Vector{Int}
    size_unit::SizeUnit
    format::Vk.Format
    usage::Vk.ImageUsageFlag
    aspect::Vk.ImageAspectFlag
end

struct BufferResourceInfo <: ResourceInfo

end

struct AttachmentResourceInfo <: ResourceInfo
    image_info::ImageResourceInfo
    attachment_type
end

struct SynchronizationRequirements
    stages::Vk.PipelineStageFlag
    access::Vk.AccessFlag
    wait_semaphores::Vector{Semaphore}
end

struct RenderGraph
    dependency_graph::SimpleDiGraph{Int}
    passes::Vector{Pass}
end

"""
Submit rendering commands to a device.

A command buffer are recorded, which may be split into multiple ones to take advantage of multithreading,
and is then submitted them to the provided device.
A fence and/or a semaphore can be provided to synchronize with the application or other commands.
"""
function render!(device, rg::RenderGraph; fence = nothing, semaphore = nothing)
    cb = get_command_buffer(device, rg)
    @record cb begin
        for pass in sort_passes(rg)
            synchronize_before(rg, pass)
            begin(cb, pass)
            record(cb, pass)
            Vk.cmd_end_render_pass()
            synchronize_after(rg, pass)
        end
    end
    submit_info = Vk.SubmitInfo([], [cb], isnothing(semaphore) ? [semaphore] : [])
    Vk.queue_submit(device, submit_info)
end

function sort_passes(rg::RenderGraph)
    indices = topological_sort_by_dfs(rg.dependency_graph)
    rg.passes[indices]
end
