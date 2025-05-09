@bitmask MemoryAccess::UInt32 begin
  NO_ACCESS = 0
  READ = 1
  WRITE = 2
end

struct Attachment
  view::ImageView
  access::MemoryAccess
end

@forward_methods Attachment field = :view begin
  Subresource
  layer_range
  mip_range
  aspect_flags
  image_format
  image_layout
  samples
  dimensions
  image_dimensions
  attachment_dimensions
  get_image
  Vk.Offset3D
  Vk.Extent3D
  Vk.set_debug_name(_, name)
end

function Base.similar(attachment::Attachment; access = attachment.access, view_kwargs...)
  Attachment(similar(attachment.view; view_kwargs...), access)
end

function load_op(access::MemoryAccess, clear::Bool)
  clear && return Vk.ATTACHMENT_LOAD_OP_CLEAR
  (READ in access || (WRITE in access && !clear)) && return Vk.ATTACHMENT_LOAD_OP_LOAD
  Vk.ATTACHMENT_LOAD_OP_DONT_CARE
end

function store_op(access::MemoryAccess)
  WRITE in access && return Vk.ATTACHMENT_STORE_OP_STORE
  Vk.ATTACHMENT_STORE_OP_DONT_CARE
end

struct ClearValue
  data::Union{Float32, UInt32, Tuple{Optional{Union{Float32, N0f8}}, Optional{UInt32}}, NTuple{4, Union{Float32, UInt32, Int32}}}
end
ClearValue(data::AbstractFloat) = ClearValue(convert(Float32, data))
ClearValue(data::Integer) = ClearValue(convert(UInt32, data))
ClearValue(data::NTuple{4,T}) where {T} = ClearValue(ntuple(i -> convert(color_clear_value_eltype(T), data[i]), 4))

color_clear_value_eltype(::Type{T}) where {T<:Signed} = Int32
color_clear_value_eltype(::Type{T}) where {T<:Unsigned} = UInt32
color_clear_value_eltype(::Type{T}) where {T<:AbstractFloat} = Float32

function Vk.ClearValue(clear::ClearValue)
  isa(clear.data, Float32) && return Vk.ClearValue(Vk.ClearDepthStencilValue(clear.data, zero(UInt32)))
  isa(clear.data, UInt32) && return Vk.ClearValue(Vk.ClearDepthStencilValue(0f0, clear.data))
  length(clear.data) == 2 && return Vk.ClearValue(Vk.ClearDepthStencilValue(something.(clear.data, 0)...))
  Vk.ClearValue(Vk.ClearColorValue(clear.data))
end

const DEFAULT_CLEAR_VALUE = ClearValue((0.0f0, 0.0f0, 0.0f0, 0.0f0))

clear_value(value::ClearValue) = Vk.ClearValue(value)
clear_value(::Nothing) = Vk.ClearValue(DEFAULT_CLEAR_VALUE)
