mutable struct CacheDiagnostics
  @atomic hits::Int
  @atomic misses::Int
end
CacheDiagnostics() = CacheDiagnostics(0, 0)

mutable struct Cache{D}
  const d::D
  @atomic diagnostics::CacheDiagnostics
end
Cache{D}() where {D} = Cache{D}(D(), CacheDiagnostics())

Base.deepcopy_internal(cache::Cache, ::IdDict) = typeof(cache)(deepcopy(cache.d), deepcopy(cache.diagnostics))

function Base.empty!(cache::Cache)
  empty!(cache.d)
  @atomic :monotonic cache.diagnostics = CacheDiagnostics()
  cache
end

@forward Cache.d (Base.haskey, Base.iterate, Base.length, Base.keys, Base.getindex)

function Base.get!(f, cache::Cache, key)
  if haskey(cache, key)
    @atomic :monotonic cache.diagnostics.hits += 1
    cache.d[key]
  else
    @atomic :monotonic cache.diagnostics.misses += 1
    get!(f, cache.d, key)
  end
end

function Base.get(f, cache::Cache, key)
  if haskey(cache, key)
    @atomic :monotonic cache.diagnostics.hits += 1
    cache[key]
  else
    @atomic :monotonic cache.diagnostics.misses += 1
    f()
  end
end

Base.get!(cache::Cache, key, default) = get!(() -> default, cache, key)
Base.get(cache::Cache, key, default) = get(() -> default, cache, key)
