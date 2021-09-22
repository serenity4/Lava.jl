struct HashTable{T}
    table::Dictionary{UInt,T}
end

@forward HashTable.table Base.get, Base.getindex, Base.setindex!, Base.isempty, Base.insert!

HashTable{T}() where {T} = HashTable{T}(Dictionary())

"""
    create_new_entry!(ht, device, info)
"""
function create_new_entry! end

function Base.get!(ht::HashTable, info, device)
    h = hash(info)
    val = get(ht, h, nothing)
    if !isnothing(val)
        return val
    else
        create_new_entry!(ht, device, info)
        ht[h]
    end
end

"""
Insert objects into the hash table by calling `f` on info arguments that were not already cached.
"""
function batch_create!(f, ht::HashTable, infos)
    hashes = hash.(infos)
    uncached = findall(Base.Fix1(!haskey, ht.table), hashes)
    if !isempty(uncached)
        objs = f(infos[uncached])
        foreach(zip(uncached, objs)) do (idx, obj)
            insert!(ht.table, hashes[idx], obj)
        end
    end
end
