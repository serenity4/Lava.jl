const LayerRange = UnitRange{Int64}
const MipRange = UnitRange{Int64}

struct Subresource
  aspect::Optional{Vk.ImageAspectFlag}
  layer_range::LayerRange
  mip_range::MipRange
end

Subresource(layer_range, mip_range) = Subresource(nothing, layer_range, mip_range)
Subresource(aspect::Optional{Vk.ImageAspectFlag} = nothing) = Subresource(aspect, 1:1, 1:1)

function Vk.ImageSubresourceRange(subresource::Subresource)
  mip_range, layer_range, aspect = Lava.mip_range(subresource), Lava.layer_range(subresource), subresource.aspect::Vk.ImageAspectFlag
  Vk.ImageSubresourceRange(aspect, mip_range[begin] - 1, 1 + mip_range[end] - mip_range[begin], layer_range[begin] - 1, 1 + layer_range[end] - layer_range[begin])
end
Base.convert(::Type{Vk.ImageSubresourceRange}, subresource::Subresource) = Vk.ImageSubresourceRange(subresource)

function Vk.ImageSubresourceLayers(subresource::Subresource)
  mip_range, layer_range, aspect = Lava.mip_range(subresource), Lava.layer_range(subresource), subresource.aspect::Vk.ImageAspectFlag
  length(mip_range) == 1 || error("Only a single mip level is allowed; for multiple mip levels, a `Vk.ImageSubresourceRange` structure will be expected")
  Vk.ImageSubresourceLayers(aspect, mip_range[begin] - 1, layer_range[begin] - 1, 1 + layer_range[end] - layer_range[begin])
end
Base.convert(::Type{Vk.ImageSubresourceLayers}, subresource::Subresource) = Vk.ImageSubresourceLayers(subresource)

function Vk.ImageSubresource(subresource::Subresource)
  mip_range, layer_range, aspect = Lava.mip_range(subresource), Lava.layer_range(subresource), subresource.aspect::Vk.ImageAspectFlag
  length(mip_range) == 1 || error("Only a single mip level is allowed; for multiple mip levels, a `Vk.ImageSubresourceRange` structure will be expected")
  length(layer_range) == 1 || error("Only a single layer is allowed; for multiple array layer_range, a `Vk.ImageSubresourceLayers` structure will be expected")
  Vk.ImageSubresource(aspect, mip_range[begin] - 1, layer_range[begin] - 1)
end
Base.convert(::Type{Vk.ImageSubresource}, subresource::Subresource) = Vk.ImageSubresource(subresource)

aspect_flags(sub::Subresource) = sub.aspect
layer_range(sub::Subresource) = sub.layer_range
mip_range(sub::Subresource) = sub.mip_range

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

Furthermore, retrievals will be much faster if the image is always used as a whole, regardless of the number of layer_range and mip levels;
whereas uses of various disjoint subresources will take more time and storage to keep track of.

You can set and retrieve information using `setindex!` and `getindex`; however, the use of `getindex` requires an identical subresource
as one used by `setindex!`, and that subresource must not have been affected by a later `setindex!` on an overlapping subresource.

To retrieve information on any subresource, regardless of what was previously set, see [`query_subresource`](@ref) and [`match_subresource`](@ref).
"""
mutable struct SubresourceMap{T}
  "The number of layer_range the image possesses."
  layer_range::Int64
  "The number of mip levels contained in the image."
  mip_range::Int64
  "Value for the whole image; `nothing` if not all subresources have the same value."
  value::Optional{T}
  "The last aspect the image was used with."
  last_aspect::Optional{Vk.ImageAspectFlag}
  value_per_aspect::Optional{Dictionary{Vk.ImageAspectFlag, SubresourceMap{T}}}
  value_per_layer::Optional{ValuePerLayer{T}}
  value_per_mip_level::Optional{ValuePerMipLevel{T}}
end

SubresourceMap{T}(layer_range, mip_range, default) where {T} = SubresourceMap{T}(layer_range, mip_range, convert(T, default), nothing, nothing, nothing, nothing)
SubresourceMap(layer_range, mip_range, default) = SubresourceMap{typeof(default)}(layer_range, mip_range, default)

layer_range(map::SubresourceMap) = 1:map.layer_range
mip_range(map::SubresourceMap) = 1:map.mip_range

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

@inline function check_layers(map::SubresourceMap, subresource::Subresource)
  @boundscheck issubset(layer_range(subresource), layer_range(map)) || throw(InvalidLayerRange(layer_range(map), layer_range(subresource)))
end

@inline function check_mip_levels(map::SubresourceMap, subresource::Subresource)
  @boundscheck issubset(mip_range(subresource), mip_range(map)) || throw(InvalidMipRange(mip_range(map), mip_range(subresource)))
end

@inline function check_subresource(map::SubresourceMap, subresource::Subresource)
  check_layers(map, subresource)
  check_mip_levels(map, subresource)
end

function Base.getindex(map::SubresourceMap{T}, subresource::Subresource) where {T}
  check_subresource(map, subresource)
  if !isnothing(map.value)
    map.value
  elseif isused(map.value_per_aspect)
    map.value_per_aspect[subresource.aspect][subresource]
  elseif isused(map.value_per_layer)
    d = map.value_per_layer[layer_range(subresource)]
    d[mip_range(subresource)]
  elseif isused(map.value_per_mip_level)
    map.value_per_mip_level[mip_range(subresource)]
  else
    error_map_invalid_state(map)
  end::T
end

function Base.getindex(map::SubresourceMap{T}) where {T}
  result = Ref{Optional{T}}(nothing)
  match_subresource(map, Subresource(1:map.layer_range, 1:map.mip_range)) do matched_layer_range, matched_mip_range, value
    isnothing(result[]) || result[] == value || error("Different subresources have different values (found $(result[]) ≠ $value); use `match_subresource` or `query_subresource` instead.")
    result[] = value
  end
  isnothing(result[]) && error_map_invalid_state(map)
  result[]::T
end

"""
    match_subresource(f, map, subresource)

Iterate through all subresource entries in `map` that are contained in `subresource`,
and call `f(matched_layer_range, matched_mip_range, value)` on each entry.
"""
function match_subresource(f::F, map::SubresourceMap{T}, subresource::Subresource) where {F,T}
  check_subresource(map, subresource)
  if !isnothing(map.value)
    f(layer_range(subresource), mip_range(subresource), map.value::T)
  elseif isused(map.value_per_aspect)
    match_subresource(f, map.value_per_aspect[subresource.aspect], subresource)
  elseif isused(map.value_per_layer)
    match_subresource_layers(f, map, subresource)
  elseif isused(map.value_per_mip_level)
    match_subresource_mip_levels(f, map, subresource)
  else
    error("$map is in an invalid state; please file an issue")
  end
  nothing
end

function match_subresource_layers(f::F, map::SubresourceMap, subresource::Subresource) where {F}
  d = get(map.value_per_layer, layer_range(subresource), nothing)
  !isnothing(d) && return match_subresource_mip_levels(f, d, subresource, layer_range(subresource))
  prev_range = 1:1
  prev_d = nothing
  for i in layer_range(subresource)
    for (range, d) in pairs(map.value_per_layer)
      in(i, range) || continue
      same_d = prev_d == d
      if !isnothing(prev_d) && same_d
        prev_range = prev_range[begin]:i
      else
        !isnothing(prev_d) && !same_d && match_subresource_mip_levels(f, prev_d, subresource, prev_range)
        prev_range = i:i
        prev_d = d
      end
      break
    end
  end
  !isnothing(prev_d) && match_subresource_mip_levels(f, prev_d, subresource, prev_range)
  nothing
end

match_subresource_mip_levels(f::F, map::SubresourceMap, subresource::Subresource) where {F} = match_subresource_mip_levels(f, map.value_per_mip_level, subresource, layer_range(map))

function match_subresource_mip_levels(f::F, d, subresource::Subresource, layer_range::LayerRange) where {F}
  value = get(d, mip_range(subresource), nothing)
  !isnothing(value) && return f(layer_range, mip_range(subresource), value)
  prev_range = 1:1
  prev_value = nothing
  for i in mip_range(subresource)
    for (range, value) in pairs(d)
      in(i, range) || continue
      if !isnothing(prev_value) && prev_value === value
        prev_range = prev_range[begin]:i
      else
        !isnothing(prev_value) && prev_value !== value && f(layer_range, prev_range, prev_value)
        prev_range = i:i
        prev_value = value
      end
      break
    end
  end
  !isnothing(prev_value) && f(layer_range, prev_range, prev_value)
  nothing
end

"""
    query_subresource(map, subresource)

Collect all subresource entries in `map` that are contained in `subresource`, returning
a vector of `(matched_layer_range, matched_mip_range) => value` pairs.
"""
function query_subresource(map::SubresourceMap{T}, subresource::Subresource) where {T}
  ret = Pair{Tuple{LayerRange, MipRange}, T}[]
  match_subresource(map, subresource) do matched_layer_range, matched_mip_range, value
    push!(ret, (matched_layer_range, matched_mip_range) => value)
  end
  ret
end

function Base.setindex!(map::SubresourceMap{T}, value::T, subresource::Subresource) where {T}
  check_subresource(map, subresource)
  if !isnothing(map.last_aspect) && !isnothing(subresource.aspect) && map.last_aspect !== subresource.aspect
    aspect_map = find_aspect_map!(map, subresource.aspect)
    aspect_map[subresource] = value
    aspect_map.last_aspect = nothing
    return map
  else
    map.value_per_aspect = nothing
    map.last_aspect = subresource.aspect
  end

  if layer_range(subresource) == layer_range(map) && mip_range(subresource) == mip_range(map)
    !isnothing(map.value_per_layer) && empty!(map.value_per_layer)
    !isnothing(map.value_per_mip_level) && empty!(map.value_per_mip_level)
    map.value = value
  elseif layer_range(subresource) == layer_range(map)
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
  dict = @something(map.value_per_mip_level, ValuePerMipLevel{T}([mip_range(map)], [map.value::T]))
  map.value = nothing
  map.value_per_mip_level = nothing
  insert!(map.value_per_layer, layer_range(map), dict)
  map
end

function update_for_layers!(map::SubresourceMap{T}, subresource::Subresource, value::T) where {T}
  parametrize_by_layer!(map)
  merge_for_range!(map.value_per_layer, layer_range(subresource)) do value_per_mip_level::Optional{ValuePerMipLevel{T}}
    if isnothing(value_per_mip_level)
      dict = ValuePerMipLevel{T}()
      replace_for_range!(dict, mip_range(subresource), value)
    else
      # Reuse dictionary if no change occurs.
      # This will also avoid splitting a key since identity (`===`) is maintained.
      # Otherwise make a new copy of it and perform the changes.
      dict = value_per_mip_level
      is_same = false
      match_range(dict, mip_range(subresource)) do other
        is_same |= other === value
      end
      if !is_same
        dict = deepcopy(dict)
        replace_for_range!(dict, mip_range(subresource), value)
      end
    end
    dict
  end
  map
end

function parametrize_by_mip_level!(map::SubresourceMap{T}) where {T}
  isused(map.value_per_layer) && return map
  isused(map.value_per_mip_level) && return map
  map.value_per_mip_level = @something(map.value_per_mip_level, ValuePerMipLevel{T}())
  insert!(map.value_per_mip_level, mip_range(map), map.value::T)
  map.value = nothing
  map
end

function update_for_mip_levels!(map::SubresourceMap{T}, subresource::Subresource, value::T) where {T}
  parametrize_by_mip_level!(map)
  replace_for_range!(map.value_per_mip_level, subresource.mip_range, value)
  map
end

"""
Update a dictionary entry (keyed by `UnitRange{Int64}`) to contain `value` at the assigned `range`.

Existing entries will be updated whenever necessary to ensure that the keys remain disjoint.
"""
replace_for_range!(dict, range::UnitRange{Int64}, value) = merge_for_range!(old -> value, dict, range)

"Call `f(value)` for each entry of `dict` that overlaps with `range`."
function match_range(f, dict, range::UnitRange{Int64})
  current_value = nothing
  for (key, value) in pairs(dict)
    isdisjoint(key, range) && continue
    !isnothing(current_value) && current_value !== value && f(current_value)
    current_value = value
  end
  !isnothing(current_value) && f(current_value)
end

"""
Replace any entry in `dict` overlapping with `range` by `f(other)`.

`other === nothing` if the operation is a sort of `insert!`; that is, if overlapping entries are strict subsets of `range`,
or if no entry exists which overlaps with `range`.
"""
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

    # Avoid splitting a key if no value changes occurs.
    new = merge(other)
    new === other && (issubset(range, key) ? (return dict) : continue)

    before, after = split_range(key, range)
    delete!(dict, key)
    !isempty(before) && insert!(dict, before, other)
    !isempty(after) && insert!(dict, after, other)
    if issubset(range, key)
      common = intersect(range, key)
      insert!(dict, common, new)
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

function find_aspect_map!(map::SubresourceMap{T}, aspect::Vk.ImageAspectFlag) where {T}
  if isnothing(map.value_per_aspect)
    map.value_per_aspect = Dictionary{Vk.ImageAspectFlag, SubresourceMap{T}}()
    aspect_map = SubresourceMap{T}(map.layer_range, map.mip_range, map.value)
    insert!(map.value_per_aspect, aspect, aspect_map)
    return aspect_map
  end

  for (known, aspect_map) in pairs(map.value_per_aspect)
    aspect in known && return aspect_map
    @assert !any(in(known), enabled_flags(aspect))
  end

  aspect_map = SubresourceMap{T}(map.layer_range, map.mip_range, map.value)
  insert!(map.value_per_aspect, aspect, aspect_map)
  aspect_map
end
