struct DescriptorSet <: LavaAbstraction
    handle::Vk.DescriptorSet
    layout::Vk.DescriptorSetLayout
end

Base.@kwdef struct ResourceMetaConfig
    textures::Int = 2048
    storage_images::Int = 512
    uniforms::Int = 4192
    storage_buffers::Int = 512
    samplers::Int = 2048
end

struct GlobalDescriptor{T}
    descriptors::Vector{T}
    names::Dictionary{Symbol,Int}
end

GlobalDescriptor{T}() where {T} = GlobalDescriptor{T}([], Dictionary())
GlobalDescriptor() = GlobalDescriptor{Any}()

function Base.convert(::Type{GlobalDescriptor{T}}, gdesc::GlobalDescriptor) where {T}
    GlobalDescriptor{T}(gdesc.descriptors, gdesc.names)
end

struct ResourceDescriptors
    textures::GlobalDescriptor{ImageBlock}
    storage_images::GlobalDescriptor{ImageBlock}
    uniforms::GlobalDescriptor{BufferBlock}
    storage_buffers::GlobalDescriptor{BufferBlock}
    samplers::GlobalDescriptor{Vk.Sampler}
    pool::Vk.DescriptorPool
    set::DescriptorSet
end

ResourceDescriptors(pool, set) = ResourceDescriptors(GlobalDescriptor(), GlobalDescriptor(), GlobalDescriptor(), GlobalDescriptor(), GlobalDescriptor(), pool, set)

function ResourceDescriptors(device, config::ResourceMetaConfig = ResourceMetaConfig())
    pool = Vk.DescriptorPool(device, 1, [
        Vk.DescriptorPoolSize(Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1),
        Vk.DescriptorPoolSize(Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE, 1),
        Vk.DescriptorPoolSize(Vk.DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1),
        Vk.DescriptorPoolSize(Vk.DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
        Vk.DescriptorPoolSize(Vk.DESCRIPTOR_TYPE_SAMPLER, 1),
    ])

    layout = Vk.DescriptorSetLayout(device, [
        Vk.DescriptorSetLayoutBinding(0, Vk.DESCRIPTOR_TYPE_SAMPLED_IMAGE, Vk.SHADER_STAGE_ALL; descriptor_count = config.textures),
        Vk.DescriptorSetLayoutBinding(1, Vk.DESCRIPTOR_TYPE_STORAGE_IMAGE, Vk.SHADER_STAGE_ALL; descriptor_count = config.storage_images),
        Vk.DescriptorSetLayoutBinding(2, Vk.DESCRIPTOR_TYPE_UNIFORM_BUFFER, Vk.SHADER_STAGE_ALL; descriptor_count = config.uniforms),
        Vk.DescriptorSetLayoutBinding(3, Vk.DESCRIPTOR_TYPE_STORAGE_BUFFER, Vk.SHADER_STAGE_ALL; descriptor_count = config.storage_buffers),
        Vk.DescriptorSetLayoutBinding(4, Vk.DESCRIPTOR_TYPE_SAMPLER, Vk.SHADER_STAGE_ALL; descriptor_count = config.samplers),
    ])

    set = DescriptorSet(first(unwrap(Vk.allocate_descriptor_sets(device, Vk.DescriptorSetAllocateInfo(pool, [layout])))), layout)
    ResourceDescriptors(pool, set)
end
