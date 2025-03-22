struct HashTable{T}
  table::LRU{UInt64,T}
end

@forward_interface HashTable field = :table interface = dict omit = [get!, get]

HashTable{T}(; maxsize = 500) where {T} = HashTable{T}(LRU{UInt64, T}(; maxsize))

Base.get(@specialize(f), ht::HashTable, info) = get(f, ht.table, info)
Base.get(ht::HashTable, info, default) = get(ht.table, info, default)

function Base.get!(@specialize(f), ht::HashTable, info)
  h = hash(info)
  val = get(ht, h, nothing)
  if !isnothing(val)
    return val
  else
    entry = f(info)
    ht[h] = entry
    entry
  end
end

"""
Insert objects into the hash table by calling `f` on info arguments that were not already cached.
"""
function batch_create!(@specialize(f), ht::HashTable, infos)
  uncached = Int[]
  hashes = UInt64[]
  for (i, info) in enumerate(infos)
    h = hash(info)
    push!(hashes, h)
    !haskey(ht.table, h) && push!(uncached, i)
  end
  if !isempty(uncached)
    objs = f(infos[uncached])
    for (idx, obj) in zip(uncached, objs)
      ht.table[hashes[idx]] = obj
    end
  end
  nothing
end
