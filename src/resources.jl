const ResourceUUID = UUID

uuid() = uuid1()
uuid(id::UUID) = id
uuid(x) = x.uuid

Vk.@bitmask_flag ResourceType::UInt32 begin
  RESOURCE_TYPE_VERTEX_BUFFER = 1
  RESOURCE_TYPE_INDEX_BUFFER = 2
  RESOURCE_TYPE_COLOR_ATTACHMENT = 4
  RESOURCE_TYPE_DEPTH_ATTACHMENT = 8
  RESOURCE_TYPE_STENCIL_ATTACHMENT = 16
  RESOURCE_TYPE_INPUT_ATTACHMENT = 32
  RESOURCE_TYPE_TEXTURE = 64
  RESOURCE_TYPE_BUFFER = 128
  RESOURCE_TYPE_IMAGE = 256
  RESOURCE_TYPE_DYNAMIC = 512
  RESOURCE_TYPE_STORAGE = 1024
  RESOURCE_TYPE_TEXEL = 2048
  RESOURCE_TYPE_UNIFORM = 4096
  RESOURCE_TYPE_SAMPLER = 8192
end

include("resources/logical.jl")
include("resources/physical.jl")
include("resources/usage.jl")

function Base.delete!(x::Union{LogicalResources,PhysicalResources,ResourceUses}, uuid::ResourceUUID)
  haskey(x.buffers, uuid) && return delete!(x.buffers, uuid)
  haskey(x.images, uuid) && return delete!(x.images, uuid)
  haskey(x.attachments, uuid) && return delete!(x.attachments, uuid)
end

function Base.getindex(x::Union{LogicalResources,PhysicalResources,ResourceUses}, uuid::ResourceUUID)
  haskey(x.buffers, uuid) && return x.buffers[uuid]
  haskey(x.images, uuid) && return x.images[uuid]
  haskey(x.attachments, uuid) && return x.attachments[uuid]
end
