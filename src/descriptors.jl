struct DescriptorSet <: LavaAbstraction
    handle::Vk.DescriptorSet
    layout::Vk.DescriptorSetLayout
end

Base.@kwdef struct GlobalDescriptorsConfig
    textures::Int = 2048
    storage_images::Int = 512
    samplers::Int = 2048
end

struct DescriptorSetBindingState{T}
    type::Vk.DescriptorType
    elements::Vector{T}
end

struct GlobalDescriptorSet
    set::DescriptorSet
    state::Dictionary{Vk.DescriptorType, Vector{Vk.DescriptorImageInfo}}
end

struct ResourceDescriptors
    pool::Vk.DescriptorPool
    gset::GlobalDescriptorSet
end

function ResourceDescriptors(device, config::GlobalDescriptorsConfig = GlobalDescriptorsConfig())
    pool = Vk.DescriptorPool(device, 1, [
        Vk.DescriptorPoolSize(Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1),
        Vk.DescriptorPoolSize(Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE, 1),
        Vk.DescriptorPoolSize(Vk.DESCRIPTOR_TYPE_SAMPLER, 1),
    ])

    layout = Vk.DescriptorSetLayout(device, [
        Vk.DescriptorSetLayoutBinding(0, Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE, Vk.SHADER_STAGE_ALL; descriptor_count = config.textures),
        Vk.DescriptorSetLayoutBinding(1, Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE, Vk.SHADER_STAGE_ALL; descriptor_count = config.storage_images),
        Vk.DescriptorSetLayoutBinding(2, Vk.DESCRIPTOR_TYPE_SAMPLER, Vk.SHADER_STAGE_ALL; descriptor_count = config.samplers),
        Vk.DescriptorSetLayoutBinding(3, Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, Vk.SHADER_STAGE_ALL; descriptor_count = config.textures),
    ], next = Vk.DescriptorSetLayoutBindingFlagsCreateInfo(repeat([Vk.DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT], 4)))

    set = DescriptorSet(first(unwrap(Vk.allocate_descriptor_sets(device, Vk.DescriptorSetAllocateInfo(pool, [layout])))), layout)
    gset = GlobalDescriptorSet(set, dictionary([
        Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE => [],
        Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE => [],
        Vk.DESCRIPTOR_TYPE_SAMPLER => [],
        Vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER => [],
    ]))
    ResourceDescriptors(pool, gset)
end
