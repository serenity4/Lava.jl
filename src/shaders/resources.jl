abstract type ShaderResource end

struct SampledImage <: ShaderResource
    image::Allocated{Created{Image,ImageCreateInfo},DeviceMemory}
    view::Allocated{Created{ImageView,ImageViewCreateInfo},DeviceMemory}
    sampler::Sampler
end

Vulkan.DescriptorType(resource::ShaderResource) = DescriptorType(typeof(resource))
Vulkan.DescriptorType(::Type{SampledImage}) = DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER

function Vulkan.WriteDescriptorSet(descriptor::Descriptor, resource::SampledImage)
    WriteDescriptorSet(
        descriptor.set,
        descriptor.binding,
        descriptor.index - 1, # 1 to 0-based indexing
        DescriptorType(resource),
        [DescriptorImageInfo(resource.sampler, resource.view, IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)],
        [],
        [],
    )
end

struct StorageBuffer <: ShaderResource
    buffer::Allocated{Buffer,DeviceMemory}
end

Vulkan.DescriptorType(::Type{StorageBuffer}) = DESCRIPTOR_TYPE_STORAGE_BUFFER
