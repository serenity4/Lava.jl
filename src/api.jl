Base.map(range::MemoryRange) = map(range.memory, offset(range), size(range))

unmap(range::MemoryRange) = unmap(range.memory)
