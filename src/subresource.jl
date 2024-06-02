const LayerRange = UnitRange{Int64}
const MipRange = UnitRange{Int64}

struct Subresource
  aspect::Optional{Vk.ImageAspectFlag}
  layers::LayerRange
  mip_levels::MipRange
end

Subresource(layers, mip_levels) = Subresource(nothing, layers, mip_levels)
Subresource(aspect::Optional{Vk.ImageAspectFlag} = nothing) = Subresource(aspect, 1:1, 1:1)

layers(sub::Subresource) = sub.layers
mip_levels(sub::Subresource) = sub.mip_levels

const ValuePerMipLevel{T} = Dictionary{MipRange, T}
const ValuePerLayer{T} = Dictionary{LayerRange, Union{T, ValuePerMipLevel{T}}}

"""
Stores information for each part (subresource) of an image.

This data structure was designed so that the most common images require low storage
and low computational complexity to encode and retrieve subresource-dependent information.

For example, an image used with a single aspect, a single layer and a single mip level
only carries a single image layout with it: there exists no subresource other than the whole image itself.

The assumption made here is that images will have, in decreasing order of probability:
- A single aspect.
- A single layer.
- A single mip level.

The highest storage and computation requirement will be for images with more than one aspect, layer *and* mip level,
unlike images with a single aspect and a single layer but several mip levels, which are fairly common and optimized for.

Furthermore, retrievals will be much faster if the image is always used as a whole, regardless of the number of layers and mip levels;
whereas uses of various disjoint subresources will take more time and storage to keep track of.

You can set and retrieve information using `setindex!` and `getindex`; however, the use of `getindex` requires an identical subresource
as one used by `setindex!`, and that subresource must not have been affected by a later `setindex!` on an overlapping subresource.

To retrieve information on any subresource, regardless of what was previously set, use `query_subresource(map)`; for point subresources
(layer and mip level ranges simultaneously of unit length), a value of type `T` is returned; for all others, either a single value of type `T`
is returned, or result of type `Dictionary{MipRange, T}` or of type `Dictionary{LayerRange, Dictionary{MipRange, T}}`
if different values map to different parts of the same subresource.
"""
mutable struct SubresourceMap{T}
  "The number of layers the image possesses."
  layers::Int64
  "The number of mip levels contained in the image."
  mip_levels::Int64
  "Value for the whole image; `nothing` if not all subresources have the same value."
  value::Optional{T}
  "The last aspect the image was used with."
  last_aspect::Optional{Vk.ImageAspectFlag}
  value_per_aspect::Optional{Dictionary{Vk.ImageAspectFlag, SubresourceMap{T}}}
  value_per_layer::Optional{ValuePerLayer{T}}
  value_per_mip_level::Optional{ValuePerMipLevel{T}}
end

SubresourceMap{T}(layers, mip_levels, default::Optional{T} = nothing) where {T} = SubresourceMap{T}(layers, mip_levels, default, nothing, nothing, nothing, nothing)

layers(map::SubresourceMap) = 1:map.layers
mip_levels(map::SubresourceMap) = 1:map.mip_levels

struct InvalidMipRange <: Exception
  allowed_range::MipRange
  provided_range::MipRange
end

Base.showerror(io::IO, exc::InvalidMipRange) = print(io, styled"InvalidMipRange: The provided mip range {red:$(exc.provided_range)} is not contained within the map's mip range {green:$(exc.allowed_range)}")

struct InvalidLayerRange <: Exception
  allowed_range::LayerRange
  provided_range::LayerRange
end

Base.showerror(io::IO, exc::InvalidLayerRange) = print(io, styled"InvalidLayerRange: The provided layer range {red:$(exc.provided_range)} is not contained within the map's layer range {green:$(exc.allowed_range)}")

Base.eltype(map::SubresourceMap{T}) where {T} = T

isused(d) = !isnothing(d) && !isempty(d)

error_map_invalid_state(map::SubresourceMap) = error("$map is in an invalid state; please file an issue")

function Base.getindex(map::SubresourceMap{T}, subresource::Subresource) where {T}
  if !isnothing(map.value)
    map.value
  elseif isused(map.value_per_aspect)
    map.value_per_aspect[subresource.aspect][subresource]
  elseif isused(map.value_per_layer)
    @boundscheck issubset(layers(subresource), layers(map)) || throw(InvalidLayerRange(layers(map), layers(subresource)))
    d = map.value_per_layer[layers(subresource)]
    @boundscheck issubset(mip_levels(subresource), mip_levels(map)) || throw(InvalidMipRange(mip_levels(map), mip_levels(subresource)))
    d[mip_levels(subresource)]
  elseif isused(map.value_per_mip_level)
    @boundscheck issubset(mip_levels(subresource), mip_levels(map)) || throw(InvalidMipRange(mip_levels(map), mip_levels(subresource)))
    map.value_per_mip_level[mip_levels(subresource)]
  else
    error_map_invalid_state(map)
  end::T
end

function query_subresource(map::SubresourceMap{T}, subresource::Subresource) where {T}
  if !isnothing(map.value)
    map.value::T
  elseif isused(map.value_per_aspect)
    query_subresource(map.value_per_aspect[subresource.aspect], subresource)
  elseif isused(map.value_per_layer)
    @boundscheck issubset(layers(subresource), layers(map)) || throw(InvalidLayerRange(layers(map), layers(subresource)))
    @boundscheck issubset(mip_levels(subresource), mip_levels(map)) || throw(InvalidMipRange(mip_levels(map), mip_levels(subresource)))
    query_subresource_layers(map, subresource)#::Union{Pair{MipRange, <:T}, Vector{Pair{MipRange, <:T}}, Vector{Pair{LayerRange, Vector{<:Pair{MipRange, <:T}}}}}
  elseif isused(map.value_per_mip_level)
    @boundscheck issubset(mip_levels(subresource), mip_levels(map)) || throw(InvalidMipRange(mip_levels(map), mip_levels(subresource)))
    query_subresource_mip_levels(map, subresource)::Union{Pair{MipRange, <:T}, Vector{Pair{MipRange, <:T}}}
  else
    error("$map is in an invalid state; please file an issue")
  end
end

function query_subresource_layers(map::SubresourceMap{T}, subresource::Subresource) where {T}
  T′ = Union{T, Vector{Pair{MipRange, T}}}
  ret = get(map.value_per_layer, subresource.layers, nothing)
  !isnothing(ret) && return query_subresource_mip_levels(map, ret, subresource)
  for i in subresource.layers
    for (range, d) in pairs(map.value_per_layer)
      in(i, range) || continue
      value = query_subresource_mip_levels(map, d, subresource)
      isnothing(ret) ? (ret = i:i => value) : begin
        if isa(ret, Pair{LayerRange, <:T′})
          (prev_range, prev_value) = ret
          prev_value == value ? (ret = prev_range[begin]:i => value) : (ret = Pair{LayerRange, T′}[ret, i:i => value])
        elseif isa(ret, Vector{Pair{LayerRange, T′}})
          (prev_range, prev_value) = ret[end]
          prev_value == value ? (ret[end] = prev_range[begin]:i => value) : push!(ret, i:i => value)
        end
      end
      break
    end
  end
  isnothing(ret) && error_map_invalid_state(map)
  isa(ret, Pair) && return ret.second::T′
  ret#::Vector{Pair{LayerRange, T′}}
end

query_subresource_mip_levels(map::SubresourceMap, subresource::Subresource) = query_subresource_mip_levels(map, map.value_per_mip_level, subresource)

function query_subresource_mip_levels(map::SubresourceMap{T}, d, subresource::Subresource) where {T}
  ret = get(d, subresource.mip_levels, nothing)
  !isnothing(ret) && return ret
  for i in subresource.mip_levels
    for (range, value) in pairs(d)
      in(i, range) || continue
      isnothing(ret) ? (ret = i:i => value) : begin
        if isa(ret, Pair{MipRange, T})
          (prev_range, prev_value) = ret
          prev_value === value ? (ret = prev_range[begin]:i => value) : (ret = Pair{MipRange, T}[ret, i:i => value])
        else # isa(ret, Vector{Pair{MipRange, T})})
          (prev_range, prev_value) = ret[end]
          prev_value === value ? (ret[end] = prev_range[begin]:i => value) : push!(ret, i:i => value)
        end
      end
      break
    end
  end
  isnothing(ret) && error_map_invalid_state(map)
  isa(ret, Pair{MipRange, T}) && return ret.second
  ret#::Vector{Pair{MipRange, T}}}
end

function Base.setindex!(map::SubresourceMap{T}, value::T, subresource::Subresource) where {T}
  if !isnothing(map.last_aspect) && !isnothing(subresource.aspect) && map.last_aspect !== subresource.aspect
    aspect_map = find_aspect_map!(map, subresource.aspect)
    aspect_map[subresource] = value
    aspect_map.last_aspect = nothing
    return map
  else
    map.value_per_aspect = nothing
    map.last_aspect = subresource.aspect
  end

  if layers(subresource) == layers(map) && mip_levels(subresource) == mip_levels(map)
    !isnothing(map.value_per_layer) && empty!(map.value_per_layer)
    !isnothing(map.value_per_mip_level) && empty!(map.value_per_mip_level)
    map.value = value
  elseif layers(subresource) == layers(map)
    !isnothing(map.value_per_layer) && empty!(map.value_per_layer)
    update_for_mip_levels!(map, subresource, value)
  else
    update_for_layers!(map, subresource, value)
  end
  map
end

function parametrize_by_layer!(map::SubresourceMap{T}) where {T}
  isused(map.value_per_layer) && return map
  map.value_per_layer = @something(map.value_per_layer, ValuePerLayer{T}())
  insert!(map.value_per_layer, layers(map), deepcopy(map.value_per_mip_level::ValuePerMipLevel{T}))
  empty!(map.value_per_mip_level)
  map
end

function update_for_layers!(map::SubresourceMap{T}, subresource::Subresource, value::T) where {T}
  parametrize_by_mip_level!(map)
  parametrize_by_layer!(map)
  merge_for_range!(map.value_per_layer, subresource.layers) do value_per_mip_level::Optional{ValuePerMipLevel{T}}
    dict = isnothing(value_per_mip_level) ? ValuePerMipLevel{T}() : deepcopy(value_per_mip_level)
    replace_for_range!(dict, subresource.mip_levels, value)
    dict
  end
  map
end

function parametrize_by_mip_level!(map::SubresourceMap{T}) where {T}
  isused(map.value_per_layer) && return map
  isused(map.value_per_mip_level) && return map
  map.value_per_mip_level = @something(map.value_per_mip_level, ValuePerMipLevel{T}())
  insert!(map.value_per_mip_level, mip_levels(map), map.value::T)
  map.value = nothing
  map
end

function update_for_mip_levels!(map::SubresourceMap{T}, subresource::Subresource, value::T) where {T}
  parametrize_by_mip_level!(map)
  replace_for_range!(map.value_per_mip_level, subresource.mip_levels, value)
  map
end

"""
Update a dictionary entry (keyed by `UnitRange{Int64}`) to contain `value` at the assigned `range`.

Existing entries will be updated whenever necessary to ensure that the keys remain disjoint.
"""
replace_for_range!(dict, range::UnitRange{Int64}, value) = merge_for_range!(old -> value, dict, range)

function merge_for_range!(merge, dict, range::UnitRange{Int64})
  # See if the range matches an existing key or supersedes any, removing all those that are superseded.
  for (key, other) in pairs(dict)
    if issubset(key, range)
      delete!(dict, key)
      if key == range
        insert!(dict, range, merge(other))
        return dict
      end
    end
  end

  for (key, other) in pairs(dict)
    isdisjoint(range, key) && continue
    before, after = split_range(key, range)
    delete!(dict, key)
    !isempty(before) && insert!(dict, before, other)
    !isempty(after) && insert!(dict, after, other)
    if !isempty(before) && !isempty(after) # also means that `range` is a strict subset of `key`
      common = intersect(range, key)
      insert!(dict, common, merge(other))
      return dict
    end
  end

  # At this point, any key overlapping with `range` will have the overlapping part deleted.
  insert!(dict, range, merge(nothing))
  dict
end

function split_range(range::UnitRange, at::UnitRange)
  i = something(findfirst(==(at[begin]), range), firstindex(range))
  j = something(findfirst(==(at[end]), range), lastindex(range))
  (range[begin:(i - 1)], range[(j + 1):end])
end

function find_aspect_map!(map::SubresourceMap, aspect::Vk.ImageAspectFlag)
  if isnothing(map.value_per_aspect)
    map.value_per_aspect = Dictionary{Vk.ImageAspectFlag, SubresourceMap}()
    aspect_map = SubresourceMap(map.layers, map.mip_levels)
    insert!(map.value_per_aspect, map, aspect_map)
    return aspect_map
  end

  for (known, aspect_map) in pairs(value_per_aspect)
    aspect in known && return aspect_map
  end

  aspect_map = SubresourceMap(map.layers, map.mip_levels)
  insert!(map.value_per_aspect, map, aspect_map)
  aspect_map
end
