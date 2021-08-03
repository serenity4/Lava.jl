abstract type Memory <: LavaAbstraction end

struct MemoryRange{M<:Memory} <: Memory
    memory::M
    offset::Int
    size::Int
end

handle(range::MemoryRange) = handle(range.memory)

offset(range::Memory) = 0
offset(range::MemoryRange) = range.offset

size(memory::MemoryRange) = memory.size
