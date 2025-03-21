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
  RESOURCE_TYPE_BUFFER
  RESOURCE_TYPE_IMAGE
  RESOURCE_TYPE_IMAGE_VIEW
  RESOURCE_TYPE_ATTACHMENT
end

ResourceID(type::ResourceType, from::ResourceID) = ResourceID(type, UInt128(from))

function ResourceID(type::ResourceType, from::UInt128 = UInt128(uuid()))
  ResourceID((from << 8) >> 8 + (UInt128(type) << 120))
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

Base.hash(r::Resource, h::UInt) = hash(r.id, h)

isnamed(r::Resource) = !isnothing(r.name)

color(r::Resource) = isattachment(r) ? 124 : isbuffer(r) ? :yellow : :cyan

function Base.show(io::IO, ::MIME"text/plain", r::Resource)
  print(io, Resource, '(')
  printstyled(io, isnamed(r) ? repr(r.name) : repr(UInt128(r.id)); color = :green)
  print(io, ",")
  color = Lava.color(r)
  islogical(r) && printstyled(io, " logical"; color)
  isimage(r) && printstyled(io, " image"; color)
  isbuffer(r) && printstyled(io, " buffer"; color)
  isimageview(r) && printstyled(io, " image view"; color)
  isattachment(r) && printstyled(io, " attachment"; color)
  print(io, ')')
end

@inline function Base.getproperty(resource::Resource, name::Symbol)
  name === :buffer && return resource.data::Buffer
  name === :image && return resource.data::Image
  name === :image_view && return resource.data::ImageView
  name === :attachment && return resource.data::Attachment
  name === :logical_buffer && return resource.data::LogicalBuffer
  name === :logical_image && return resource.data::LogicalImage
  name === :logical_image_view && return resource.data::LogicalImageView
  name === :logical_attachment && return resource.data::LogicalAttachment
  getfield(resource, name)
end

Vk.set_debug_name(resource::Resource, name) = set_debug_name(resource.data, name)

function Resource(type::ResourceType, data; name = nothing, flags = ResourceFlags(0))
  !isnothing(data) && Vk.set_debug_name(data, name)
  Resource(ResourceID(type), data, name, flags)
end
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
isimageview(x) = resource_type(x) == RESOURCE_TYPE_IMAGE_VIEW
isattachment(x) = resource_type(x) == RESOURCE_TYPE_ATTACHMENT

isphysical(resource::Resource) = !in(RESOURCE_IS_LOGICAL, resource.flags)
islogical(resource::Resource) = in(RESOURCE_IS_LOGICAL, resource.flags)
function promote_to_physical(resource::Resource, x)
  physical = setproperties(resource, (; data = x, flags = resource.flags & ~RESOURCE_IS_LOGICAL))
  set_debug_name(physical, resource.name)
  physical
end

function DeviceAddress(resource::Resource)
  isbuffer(resource) && isphysical(resource) || error("Device addresses can only be retrieved from physical buffer resources.")
  DeviceAddress(resource.data::Buffer)
end

include("resources/logical.jl")

resource_type(resource::Union{LogicalBuffer, Buffer}) = RESOURCE_TYPE_BUFFER
resource_type(resource::Union{LogicalImage, Image}) = RESOURCE_TYPE_IMAGE
resource_type(resource::Union{LogicalImageView, ImageView}) = RESOURCE_TYPE_IMAGE_VIEW
resource_type(resource::Union{LogicalAttachment, Attachment}) = RESOURCE_TYPE_ATTACHMENT

ResourceFlags(::Union{LogicalBuffer, LogicalImage, LogicalImageView, LogicalAttachment}) = RESOURCE_IS_LOGICAL
ResourceFlags(::Union{Buffer, Image, ImageView, Attachment}) = zero(ResourceFlags)

include("resources/usage.jl")

function samples(r::Resource)
  isbuffer(r) && return 1
  samples(r.data::Union{Image, LogicalImage, ImageView, LogicalImageView, Attachment, LogicalAttachment})
end

function dimensions(r::Resource)
  isbuffer(r) && error("Cannot retrieve dimensions for buffer resources.")
  dimensions(r.data::Union{Image, LogicalImage, ImageView, LogicalImageView, Attachment, LogicalAttachment})
end
function image_dimensions(r::Resource)
  isbuffer(r) && error("Cannot retrieve image dimensions for buffer resources.")
  image_dimensions(r.data::Union{Image, LogicalImage, ImageView, LogicalImageView, Attachment, LogicalAttachment})
end
function attachment_dimensions(r::Resource)
  isbuffer(r) && error("Cannot retrieve attachment dimensions for buffer resources.")
  attachment_dimensions(r.data::Union{Attachment, LogicalAttachment})
end

function image_format(r::Resource)
  isimage(r) && return image_format(r.data::Union{Image, LogicalImage})
  isimageview(r) && return image_format(r.data::Union{ImageView, LogicalImageView})
  isattachment(r) && return image_format(r.data::Union{Attachment, LogicalAttachment})
  throw(ArgumentError("Formats can only be extracted from image, image view or attachment resources."))
end

function get_image(r::Resource)
  @match resource_type(r) begin
    &RESOURCE_TYPE_IMAGE => r.image
    &RESOURCE_TYPE_IMAGE_VIEW => r.image_view.image
    &RESOURCE_TYPE_ATTACHMENT => r.attachment.view.image
    type => error("Expected image, image view or attachment, got resource of type ", type)
  end
end

Base.similar(r::Resource, args...; name = nothing, kwargs...) = Resource(similar(r.data, args...; kwargs...); name)

function Base.copyto!(to::Resource, from, device; kwargs...)
  to = to.data::Union{Buffer, Image, ImageView, Attachment}
  copyto!(to, from, device; kwargs...)
end

function Base.copyto!(to, from::Resource, device; kwargs...)
  from = from.data::Union{Buffer, Image, ImageView, Attachment}
  copyto!(to, from, device; kwargs...)
end

function Base.copyto!(to::Resource, from::Resource, device; kwargs...)
  to = to.data::Union{Buffer, Image, ImageView, Attachment}
  from = from.data::Union{Buffer, Image, ImageView, Attachment}
  copyto!(to, from, device; kwargs...)
end
