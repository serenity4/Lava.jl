Vk.device(x::LavaAbstraction) = Vk.device(handle(x))
Vk.instance(x::LavaAbstraction) = Vk.instance(handle(x))

struct VkMemory <: AbstractMemory
    handle::Vk.DeviceMemory
end

struct VkImage <: AbstractImage
    handle::Vk.Image
end

struct VkBuffer <: AbstractBuffer
    handle::Vk.Buffer
end

Vk.bind_buffer_memory(buffer::AbstractBuffer, memory) = Vk.bind_buffer_memory(device(buffer), handle(memory), offset(memory))
Vk.bind_image_memory(image::AbstractImage, memory) = Vk.bind_image_memory(device(image), handle(memory), offset(memory))

Base.bind(buffer::VkBuffer, memory::VkMemory) = bind_buffer_memory(buffer, memory)
Base.bind(image::VkImage, memory::VkMemory) = bind_image_memory(image, memory)

Base.map(memory::VkMemory, offset, size) = map_memory(device(memory), handle(memory), offset, size)
unmap(memory::VkMemory) = unmap_memory(device(memory), memory)
