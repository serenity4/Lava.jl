
abstract type Buffer <: LavaAbstraction end

abstract type DataOperation end

abstract type Transfer <: DataOperation end
abstract type VideoDecode <: DataOperation end
abstract type VideoEncode <: DataOperation end

abstract type StorageType end

abstract type Vertex <: StorageType end
abstract type Index{T} <: StorageType end

abstract type SparsityType end

abstract type SparseResidency <: SparsityType end
abstract type SparseBinding <: SparsityType end

abstract type Sparse{T<:SparsityType,O<:LavaAbstraction} <: StorageType end

struct SubBuffer{B<:Buffer} <: Buffer
    buffer::B
    offset::Int
    stride::Int
end

handle(sub::SubBuffer) = handle(sub.buffer)

offset(buffer::Buffer) = 0
offset(buffer::SubBuffer) = buffer.offset

stride(buffer::Buffer) = 0
stride(buffer::SubBuffer) = buffer.stride
