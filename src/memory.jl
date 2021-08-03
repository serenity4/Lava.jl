abstract type AbstractMemory <: LavaAbstraction end

struct MemoryRange{M<:AbstractMemory} <: AbstractMemory
    memory::M
    offset::Int
    size::Int
end

handle(range::MemoryRange) = handle(range.memory)

offset(range::AbstractMemory) = 0
offset(range::MemoryRange) = range.offset

size(memory::MemoryRange) = memory.size
