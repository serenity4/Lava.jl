uuid() = uuid1()

primitive type NodeID 128 end

NodeID(id::UInt128) = reinterpret(NodeID, id)
Base.UInt128(id::NodeID) = reinterpret(UInt128, id)
NodeID(id::UUID) = NodeID(UInt128(id))
NodeID() = NodeID(uuid())

primitive type ResourceID 128 end

ResourceID(id::UInt128) = reinterpret(ResourceID, id)
Base.UInt128(id::ResourceID) = reinterpret(UInt128, id)
ResourceID(id::UUID) = ResourceID(UInt128(id))

@enum ResourceType::UInt8 begin
  RESOURCE_TYPE_IMAGE
  RESOURCE_TYPE_BUFFER
  RESOURCE_TYPE_ATTACHMENT
end

function ResourceID(type::ResourceType)
  id = UInt128(uuid())
  ResourceID((id << 8) >> 8 + (UInt128(type) << 120))
end

@bitmask ResourceFlags::UInt32 begin
  RESOURCE_IS_LOGICAL = 1
end

struct Resource
  id::ResourceID
  data::Any
  name::Optional{Symbol}
  flags::ResourceFlags
end

isnamed(r::Resource) = !isnothing(r.name)

color(r::Resource) = isattachment(r) ? 124 : isbuffer(r) ? :yellow : :cyan

function Base.show(io::IO, ::MIME"text/plain", r::Resource)
  print(io, Resource, '(')
  printstyled(io, isnamed(r) ? repr(r.name) : r.id; color = :green)
  print(io, ", ")
  printstyled(io, islogical(r) ? "logical " : "", isimage(r) ? "image" : isbuffer(r) ? "buffer" : "attachment"; color = color(r))
  print(io, ')')
end

@inline function Base.getproperty(resource::Resource, name::Symbol)
  name === :buffer && return resource.data::Buffer
  name === :image && return resource.data::Image
  name === :attachment && return resource.data::Attachment
  name === :logical_buffer && return resource.data::LogicalBuffer
  name === :logical_image && return resource.data::LogicalImage
  name === :logical_attachment && return resource.data::LogicalAttachment
  getfield(resource, name)
end

Resource(type::ResourceType, data; name = nothing, flags = ResourceFlags(0)) = Resource(ResourceID(type), data, name, flags)
Resource(data; name = nothing, flags = zero(ResourceFlags)) = Resource(resource_type(data), data; name, flags = flags | ResourceFlags(data))

print_name(io::IO, resource::Resource) = printstyled(IOContext(io, :color => true), isnothing(resource.name) ? resource.id : resource.name; color = color(resource))

resource_type(id::ResourceID) = ResourceType(UInt8(UInt128(id) >> 120))
resource_type(resource::Resource) = resource_type(resource.id)

function assert_type(resource::Resource, rtype::ResourceType)
  @assert resource_type(resource) == rtype "Resource type is $(resource_type(resource)) (expected $rtype)"
  resource
end

isbuffer(x) = resource_type(x) == RESOURCE_TYPE_BUFFER
isimage(x) = resource_type(x) == RESOURCE_TYPE_IMAGE
isattachment(x) = resource_type(x) == RESOURCE_TYPE_ATTACHMENT

isphysical(resource::Resource) = !in(RESOURCE_IS_LOGICAL, resource.flags)
islogical(resource::Resource) = in(RESOURCE_IS_LOGICAL, resource.flags)
promote_to_physical(resource::Resource, x) = setproperties(resource, (; data = x, flags = resource.flags & ~RESOURCE_IS_LOGICAL))

function DeviceAddress(resource::Resource)
  isbuffer(resource) && isphysical(resource) || error("Device addresses can only be retrieved from physical buffer resources.")
  DeviceAddress(resource.data::Buffer)
end

include("resources/logical.jl")

resource_type(resource::Union{LogicalBuffer,Buffer}) = RESOURCE_TYPE_BUFFER
resource_type(resource::Union{LogicalImage,Image}) = RESOURCE_TYPE_IMAGE
resource_type(resource::Union{LogicalAttachment,Attachment}) = RESOURCE_TYPE_ATTACHMENT

ResourceFlags(::Union{LogicalBuffer, LogicalImage, LogicalAttachment}) = RESOURCE_IS_LOGICAL
ResourceFlags(::Union{Buffer, Image, Attachment}) = zero(ResourceFlags)

include("resources/usage.jl")

function samples(r::Resource)
  isbuffer(r) && return 1
  samples(r.data::Union{Image,LogicalImage,Attachment,LogicalAttachment})
end

function dimensions(r::Resource)
  isbuffer(r) && error("Cannot retrieve dimensions for buffer resources.")
  dimensions(r.data::Union{Image,LogicalImage,Attachment,LogicalAttachment})
end
function image_dimensions(r::Resource)
  isbuffer(r) && error("Cannot retrieve image dimensions for buffer resources.")
  image_dimensions(r.data::Union{Image,LogicalImage,Attachment,LogicalAttachment})
end
function attachment_dimensions(r::Resource)
  isbuffer(r) && error("Cannot retrieve attachment dimensions for buffer resources.")
  attachment_dimensions(r.data::Union{Attachment,LogicalAttachment})
end

function image_format(r::Resource)
  isimage(r) && return (r.data::Union{Image,LogicalImage}).format
  isattachment(r) && return islogical(r) ? r.logical_attachment.format : r.attachment.view.image.format
  throw(ArgumentError("Formats can only be extracted from image or attachment resources."))
end

Base.similar(r::Resource, args...; name = nothing, kwargs...) = Resource(similar(r.data, args...; kwargs...); name)
