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

const ResourceStorage = Union{LogicalResources,PhysicalResources,ResourceUses}

const ResourceData_T = Union{Buffer,Image,Attachment}

for R in [:Buffer, :Image, :Attachment]
  # BufferResource_T, ImageResource_T, AttachmentResource_T
  @eval const $(Symbol(R, :Resource_T)) = Union{$(Symbol(:Logical, R)),$(Symbol(:Physical, R)),$(Symbol(R, :Usage))}
  @eval const $(Symbol(R, :Any_T)) = Union{$(Symbol(R, :Resource_T)), $R}
  r = Symbol(lowercase(string(R)))
  for S in [:Logical, :Physical]
    @eval $r(x::$(Symbol(S, :Resources)), args...; kwargs...) = new!(x, $(Symbol(S, R))(uuid(), args...; kwargs...))
    @eval AbstractResourceType(::Type{$(Symbol(S, :Resources))}) = $(Symbol(S, :Resource))
    @eval $(Symbol(S, :Resource))(uuid::ResourceUUID, data::$R) = $(Symbol(S, R))(uuid, data)
  end
end

const Resource_T = Union{BufferResource_T,ImageResource_T,AttachmentResource_T}

usage(::BufferAny_T, dep::ResourceDependency) = BufferUsage(; dep.type, dep.access)
usage(::ImageAny_T, dep::ResourceDependency) = ImageUsage(; dep.type, dep.access)
usage(::AttachmentAny_T, dep::ResourceDependency) = AttachmentUsage(; dep.type, dep.access, dep.clear_value, Vk.SampleCountFlag(dep.samples))

storage_dict(x, resource) = storage_dict(x, typeof(resource))
storage_dict(x, ::Type{<:BufferAny_T}) = x.buffers
storage_dict(x, ::Type{<:ImageAny_T}) = x.images
storage_dict(x, ::Type{<:AttachmentAny_T}) = x.attachments

new!(x::Union{LogicalResources,PhysicalResources}, data::ResourceData_T) = new!(x, PhysicalResource(uuid(), data))

function new!(x::ResourceStorage, data::Resource_T)
  insert!(storage_dict(x, data), data.uuid, data)
  data
end
Base.merge(x::T, y::T) where {T<:ResourceStorage} = T((merge(storage_dict(x, T), storage_dict(y, T)) for T in (Buffer, Image, Attachment))...)
Base.getindex(x::ResourceStorage, data::Resource_T) = storage_dict(x, data)[data.uuid]
Base.in(data::Resource_T, x::ResourceStorage) = haskey(storage_dict(x, data), data.uuid)
Base.delete!(x::ResourceStorage, data::Resource_T) = delete!(storage_dict(x, data), data.uuid)
Base.insert!(x::ResourceStorage, uuid::ResourceUUID, data::Resource_T) = insert!(storage_dict(x, data), uuid, data)
Base.insert!(x::ResourceStorage, uuid::ResourceUUID, data::ResourceData_T) =
  insert!(storage_dict(x, data), uuid, AbstractResourceType(typeof(x))(uuid, data))

for f in [:getindex, :delete!]
  @eval function Base.$f(x::ResourceStorage, uuid::ResourceUUID)
    haskey(x.buffers, uuid) && return $f(x.buffers, uuid)
    haskey(x.images, uuid) && return $f(x.images, uuid)
    haskey(x.attachments, uuid) && return $f(x.attachments, uuid)
    throw(KeyError("Resource $uuid not found in $x"))
  end
end

function Base.in(uuid::ResourceUUID, x::Union{LogicalResources,PhysicalResources})
  haskey(x.buffers, uuid) || haskey(x.images, uuid) || haskey(x.attachments, uuid)
end
