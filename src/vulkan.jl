Vk.device(x::LavaAbstraction) = Vk.device(handle(x))
Vk.instance(x::LavaAbstraction) = Vk.instance(handle(x))

struct VkMemory <: Memory
    handle::Vk.DeviceMemory
end

struct VkImage{N} <: Image{N}
    handle::Vk.Image
end

struct VkBuffer <: Buffer
    handle::Vk.Buffer
end

struct VkCommandBuffer <: CommandRecord
    handle::Vk.CommandBuffer
end

struct VkPipeline{T} <: Pipeline{T}
    handle::Vk.Pipeline
end

struct VkDescriptorSet <: ResourceReference
    handle::Vk.DescriptorSet
end

struct VkCommandPool <: Pool{VkCommandBuffer}
    handle::Vk.CommandPool
end

struct VkDescriptorPool <: Pool{VkDescriptorSet}
    handle::Vk.DescriptorPool
end

struct VkFence <: SynchronizationPrimitive{CPU,GPU}
    handle::Vk.Fence
end

struct VkEvent <: SynchronizationPrimitive{Any,Any}
    handle::Vk.Event
end

struct VkSemaphore <: SynchronizationPrimitive{GPU,GPU}
    handle::Vk.Semaphore
end

Vk.bind_buffer_memory(buffer::Buffer, memory) = Vk.bind_buffer_memory(device(buffer), memory, offset(memory))
Vk.bind_image_memory(image::Image, memory) = Vk.bind_image_memory(device(image), memory, offset(memory))
Vk.cmd_bind_pipeline(cbuffer::VkCommandBuffer, pipeline::VkPipeline) = Vk.cmd_bind_pipeline(cbuffer, Vk.PipelineBindPoint(typeof(pipeline)), pipeline)
Vk.cmd_bind_vertex_buffers(cbuffer::VkCommandBuffer, buffers::AbstractVector{<:Buffer}) = Vk.cmd_bind_vertex_buffers(cbuffer, buffers, offset.(buffers))
Vk.cmd_bind_index_buffer(cbuffer::VkCommandBuffer, buffers::AbstractVector{<:Buffer}, T) = Vk.cmd_bind_index_buffer(cbuffer, buffer, offset(buffer), IndexType(T))

Base.bind(buffer::Buffer, memory::Memory) = Vk.bind_buffer_memory(buffer, memory)
Base.bind(image::Image, memory::Memory) = Vk.bind_image_memory(image, memory)
record(cbuffer::VkCommandBuffer, ::typeof(bind), pipeline::VkPipeline) = Vk.cmd_bind_pipeline(cbuffer, pipeline)
record(cbuffer::VkCommandBuffer, ::typeof(bind), buffers::AbstractVector{<:Buffer}, ::Type{<:Vertex}) = Vk.cmd_bind_vertex_buffers(cbuffer, pipeline, buffers)
record(cbuffer::VkCommandBuffer, ::typeof(bind), buffers::Buffer, ::Type{<:Index{T}}) where {T} = Vk.cmd_bind_index_buffer(cbuffer, pipeline, buffer, T)

Vk.IndexType(::Type{UInt8}) = Vk.INDEX_TYPE_UINT8_EXT
Vk.IndexType(::Type{UInt16}) = Vk.INDEX_TYPE_UINT16
Vk.IndexType(::Type{UInt32}) = Vk.INDEX_TYPE_UINT32
Vk.IndexType(::Type{Nothing}) = Vk.INDEX_TYPE_NONE_KHR

Vk.PipelineBindPoint(::Type{<:VkPipeline{<:Graphics}}) = Vk.PIPELINE_BIND_POINT_GRAPHICS
Vk.PipelineBindPoint(::Type{<:VkPipeline{<:Compute}}) = Vk.PIPELINE_BIND_POINT_COMPUTE
Vk.PipelineBindPoint(::Type{<:VkPipeline{<:RayTracing}}) = Vk.PIPELINE_BIND_POINT_RAY_TRACING_KHR

Base.reset(cbuffer::VkCommandBuffer; flags=0) = Vk.reset_command_buffer(cbuffer; flags)
Base.reset(pool::Pool{VkCommandBuffer}; flags=0) = Vk.reset_command_pool(device(pool), pool; flags)
Base.reset(pool::Pool{VkDescriptorSet}; flags=0) = Vk.reset_descriptor_pool(device(pool), pool; flags)
Base.reset(event::SynchronizationPrimitive{Any,Any}) = Vk.reset_event(device(event), fences)
Base.reset(fences::AbstractVector{<:SynchronizationPrimitive{CPU,GPU}}) = Vk.reset_fences(device(first(fences)), fences)

Base.map(memory::Memory, offset, size) = Vk.map_memory(device(memory), handle(memory), offset, size)
unmap(memory::Memory) = Vk.unmap_memory(device(memory), memory)

# # Flags

flag(::Type{Sparse{SparseBinding,VkBuffer}}) = Vk.BUFFER_CREATE_SPARSE_BINDING_BIT
flag(::Type{Sparse{SparseBinding,VkImage}}) = Vk.IMAGE_CREATE_SPARSE_BINDING_BIT

flag(::Type{Sparse{SparseResidency,VkBuffer}}) = Vk.BUFFER_CREATE_SPARSE_RESIDENCY_BIT
flag(::Type{Sparse{SparseResidency,VkImage}}) = Vk.IMAGE_CREATE_SPARSE_RESIDENCY_BIT

for N in 1:3
    @eval flag(::Type{<:Image{$N}}) = $(getproperty(Vk, Symbol("IMAGE_TYPE_", N, "D")))
    @eval flag(::Type{<:View{<:Image{$N}}}) = $(getproperty(Vk, Symbol("IMAGE_VIEW_TYPE_", N, "D")))
end
