function texture_id!(fg::FrameGraph, arg::Texture, pass)
    # let's create the resource eagerly for now
    img = image(fg, arg.name)
    (; frame, resource_graph) = fg
    if !isnothing(arg.sampling)
        # combined image sampler
        sampling = arg.sampling
        sampler = Vk.Sampler(device(fg), sampling)
        # preserve sampler
        register(frame, gensym(:sampler), sampler; persistent = false)
        combined_image_sampler_state = frame.gd.resources.gset.state[Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER]
        push!(combined_image_sampler_state, Vk.DescriptorImageInfo(sampler, ImageView(img), image_layout(resource_graph, fg.resources[arg.name], pass)))
        length(combined_image_sampler_state)
    else
        # sampled image
        sampler = empty_handle(Vk.Sampler)
        sampled_image_state = frame.gd.resources.gset.state[Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE]
        push!(sampled_image_state, Vk.DescriptorImageInfo(sampler, ImageView(img), image_layout(resource_graph, fg.resources[arg.name], pass)))
        length(sampled_image_state)
    end
end

function sampler_id!(fg::FrameGraph, arg::Sampling)
    (; frame) = fg
    view = empty_handle(Vk.ImageView)
    sampler = Vk.Sampler(device(fg), sampling)
    # preserve sampler
    register(frame, gensym(:sampler), sampler; persistent = false)
    sampler_state = frame.gd.resources.gset.state[Vk.DESCRIPTOR_TYPE_SAMPLER]
    push!(sampler_state, Vk.DescriptorImageInfo(sampler, view, Vk.IMAGE_LAYOUT_UNDEFINED))
    length(sampler_state)
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
