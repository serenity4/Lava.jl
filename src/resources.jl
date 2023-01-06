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

@bitmask_flag ResourceFlags::UInt32 begin
  RESOURCE_IS_LOGICAL = 1
end

struct Resource
  id::ResourceID
  data::Any
  flags::ResourceFlags
end

Resource(type::ResourceType, data, flags = ResourceFlags(0)) = Resource(ResourceID(type), data, flags)
Resource(data::Union{Buffer,Image,Attachment}, flags = ResourceFlags(0)) = Resource(resource_type(data), data, flags)

resource_type(id::ResourceID) = ResourceType(UInt8(UInt128(id) >> 120))
resource_type(resource::Resource) = resource_type(resource.id)
resource_type(resource::Buffer) = RESOURCE_TYPE_BUFFER
resource_type(resource::Image) = RESOURCE_TYPE_IMAGE
resource_type(resource::Attachment) = RESOURCE_TYPE_ATTACHMENT

assert_type(resource::Resource, rtype::ResourceType) = @assert resource_type(resource) == rtype "Resource type is $(resource_type(resource)) (expected $rtype)"

isbuffer(x) = resource_type(x) == RESOURCE_TYPE_BUFFER
isimage(x) = resource_type(x) == RESOURCE_TYPE_IMAGE
isattachment(x) = resource_type(x) == RESOURCE_TYPE_ATTACHMENT

isphysical(resource::Resource) = !in(RESOURCE_IS_LOGICAL, resource.flags)
islogical(resource::Resource) = in(RESOURCE_IS_LOGICAL, resource.flags)
logical_resource(type::ResourceType, data, flags = ResourceFlags(0)) = Resource(type, data, flags | RESOURCE_IS_LOGICAL)
promote_to_physical(resource::Resource, x) = setproperties(resource, (; data = x, flags = resource.flags & ~RESOURCE_IS_LOGICAL))

function DeviceAddress(resource::Resource)
  isbuffer(resource) && isphysical(resource) || error("Device addresses can only be retrieved from physical buffer resources.")
  DeviceAddress(resource.data::Buffer)
end

include("resources/logical.jl")
include("resources/usage.jl")
