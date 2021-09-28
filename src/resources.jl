"""
Frame-global structure that holds all data needed in the frame.

Its linear allocator is used for allocating lots of small objects, like material parameters and vertex data.

Other resources that require a global descriptor set (bind-once strategy) are put into a `ResourceDescriptors`.
This includes general image data & samplers, with the corresponding descriptors.

The index list is used to append index data and is turned into an index buffer before initiating the render sequence.
"""
struct GlobalData
    allocator::LinearAllocator
    resources::ResourceDescriptors
    index_list::Vector{UInt32}
    index_buffer::Ref{BufferBlock{MemoryBlock}}
end

GlobalData(device) = GlobalData(
    LinearAllocator(device, 1_000_000), # 1 MB
    ResourceDescriptors(device),
    [],
    Ref{BufferBlock{MemoryBlock}}(),
)

function texture_id!(resources::ResourceDescriptors, fg::FrameGraph, arg::Texture, pass::Symbol)
    # let's create the resource eagerly for now
    image = resources.textures[arg.name]
    if !isnothing(arg.sampling)
        # combined image sampler
        sampling = arg.sampling
        sampler = Vk.Sampler(device(fg), sampling)
        push!(resources.samplers.resources, sampler)
        combined_image_sampler_state = resources.gset.state[Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER]
        push!(combined_image_sampler_state, Vk.DescriptorImageInfo(sampler, ImageView(image), image_layout(fg, arg.name, pass)))
        length(combined_image_sampler_state)
    else
        # sampled image
        # need a sampler even though it is ignored in the struct
        # we'll have to remove this hack some day
        sampler = isempty(resources.samplers.resources) ? Vk.Sampler(device(fg), DEFAULT_SAMPLING) : first(resources.samplers.resources)
        sampled_image_state = resources.gset.state[Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE]
        push!(sampled_image_state, Vk.DescriptorImageInfo(sampler, ImageView(image), image_layout(fg, arg.name, pass)))
        length(sampled_image_state)
    end
end

function sampler_id!(resources::ResourceDescriptors, fg::FrameGraph, arg::Sampling, pass::Symbol)
    view = ImageView(first(resources.textures.resources)) # need an image view, will be ignored by Vulkan
    sampler = Vk.Sampler(device(fg), sampling)
    push!(resources.samplers.resources, sampler)
    sampler_state = resources.gset.state[Vk.DESCRIPTOR_TYPE_SAMPLER]
    push!(sampler_state, Vk.DescriptorImageInfo(sampler, view, Vk.IMAGE_LAYOUT_UNDEFINED))
    length(sampler_state)
end

function populate_descriptor_sets!(gd::GlobalData)
    state = gd.resources.gset.state
    types = [Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE, Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE, Vk.DESCRIPTOR_TYPE_SAMPLER, Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER]
    writes = map(enumerate(types)) do (i, type)
        infos = state[type]
        Vk.WriteDescriptorSet(0, 0, i - 1, length(infos), type, infos)
    end
    Vk.update_descriptor_sets(
        device(gd.resources.gset.set),
        writes,
        [],
    )
end

#=


"""
Encode a lifetime context that allows the deletion of staged resources once expired.
"""
abstract type LifeTimeContext end

"""
Period of time that extends for a whole frame.
Work between frames is assumed to be independent.
"""
struct FrameContext <: LifeTimeContext end

struct PersistentContext <: LifeTimeContext end

struct Resources
    descriptors::ResourceDescriptors

    "Number of frames during which resources were not used."
    unused::Dictionary{Int,Vector{Int}}
    "Persistent resources, never to be garbage-collected."
    persistent::Vector{Int}
end

"""
Transition from one frame to another.
"""
function transition(resources::Resources)
    for resource in resources
end

struct Frame
    fg::FrameGraph
    data::GlobalData
    resources::Resources
    fences::Vector{Vk.Fence}
end

=#
