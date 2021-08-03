
abstract type AbstractBuffer <: LavaAbstraction end

struct SubBuffer{B<:AbstractBuffer} <: AbstractBuffer
    buffer::B
    offset::Int
    stride::Int
end

handle(sub::SubBuffer) = handle(sub.buffer)

offset(buffer::AbstractBuffer) = 0
offset(buffer::SubBuffer) = buffer.offset

stride(buffer::AbstractBuffer) = 0
stride(buffer::SubBuffer) = buffer.stride
