using Lava: Subresource, SubresourceMap, replace_for_range!, InvalidMipRange, InvalidLayerRange, match_subresource, query_subresource
using Dictionaries, Test

all_keys_disjoint(dict) = all(x -> all(y -> x == y || isdisjoint(x, y), keys(dict)), keys(dict))

function validate_map(map::SubresourceMap)
  isused(d) = !isnothing(d) && !isempty(d)
  if map.value !== nothing
    @test !isused(map.value_per_aspect)
    @test !isused(map.value_per_layer)
    @test !isused(map.value_per_mip_level)
  elseif isused(map.value_per_aspect)
    @test map.last_aspect !== nothing
    @test !isused(map.value_per_layer)
    @test !isused(map.value_per_mip_level)
    @test all_keys_disjoint(map.value_per_aspect)
  elseif isused(map.value_per_layer)
    @test !isused(map.value_per_mip_level)
    @test all_keys_disjoint(map.value_per_layer)
    for entry in map.value_per_layer
      @test all_keys_disjoint(entry)
    end
  end
  nothing
end

@testset "Subresources" begin
  @testset "Updating disjoint range entries" begin
    @test all_keys_disjoint(dictionary([1:3 => :a, 4:6 => :b]))
    @test all_keys_disjoint(dictionary([]))
    @test all_keys_disjoint(dictionary([1:5 => :a]))
    @test !all_keys_disjoint(dictionary([1:3 => :a, 2:5 => :b]))

    d = dictionary([1:3 => :a, 4:6 => :b])
    @test all_keys_disjoint(d)
    replace_for_range!(d, 1:3, :c)
    @test d[1:3] == :c
    @test all_keys_disjoint(d)

    d = dictionary([1:3 => :a, 4:6 => :b])
    replace_for_range!(d, 2:3, :c)
    @test length(d) == 3
    @test d[2:3] == :c
    @test d[1:1] == :a
    @test all_keys_disjoint(d)

    d = dictionary([1:3 => :a, 4:6 => :b])
    replace_for_range!(d, 4:5, :c)
    @test length(d) == 3
    @test d[4:5] == :c
    @test d[6:6] == :b
    @test all_keys_disjoint(d)

    d = dictionary([1:3 => :a, 5:7 => :b])
    replace_for_range!(d, 1:4, :c)
    @test length(d) == 2
    @test d[1:4] == :c
    @test d[5:7] == :b

    d = dictionary([1:3 => :a, 4:6 => :b])
    replace_for_range!(d, 1:4, :c)
    @test length(d) == 2
    @test d[1:4] == :c
    @test d[5:6] == :b

    d = dictionary([1:3 => :a, 4:6 => :b])
    replace_for_range!(d, 1:8, :c)
    @test length(d) == 1
    @test d[1:8] == :c

    d = dictionary([1:3 => :a, 4:6 => :b])
    replace_for_range!(d, 2:2, :c)
    @test length(d) == 4
    @test d[1:1] == :a
    @test d[2:2] == :c
    @test d[3:3] == :a
    @test all_keys_disjoint(d)
  end

  smap = SubresourceMap(1, 1, :a)
  @test smap.value == :a
  sub = Subresource()
  smap[sub] = :a
  @test smap.value == :a
  validate_map(smap)

  smap = SubresourceMap(6, 4, :a)
  @test smap.value == :a
  sub1 = Subresource(1:6, 1:4)
  smap[sub1] = :a
  @test smap.value == :a
  smap[sub1] = :b
  @test smap.value == :b
  sub2 = Subresource(3:4, 1:4)
  smap[sub2] = :c
  @test smap[Subresource(1:2, 1:4)] == :b
  @test smap[sub2] == :c
  @test smap[Subresource(5:6, 1:4)] == :b
  validate_map(smap)
  sub3 = Subresource(2:3, 1:2)
  smap[sub3] = :d
  @test smap[sub3] == :d
  sub4 = Subresource(2:3, 3:4)
  smap[sub4] = :e
  @test smap[sub3] == :d
  @test smap[sub4] == :e
  sub5 = Subresource(5:6, 1:4)
  smap[sub5] = :f
  @test smap[sub3] == :d
  @test smap[sub4] == :e
  @test smap[sub5] == :f
  validate_map(smap)
  @test query_subresource(smap, sub5) == [(5:6, 1:4) => :f]
  @test query_subresource(smap, Subresource(2:3, 1:4)) == [(2:3, 1:2) => :d, (2:3, 3:4) => :e]
  @test query_subresource(smap, Subresource(3:3, 1:4)) == [(3:3, 1:2) => :d, (3:3, 3:4) => :e]
  @test query_subresource(smap, Subresource(1:3, 1:4)) == [(1:1, 1:4) => :b, (2:3, 1:2) => :d, (2:3, 3:4) => :e]
  @test query_subresource(smap, Subresource(1:6, 1:4)) == [(1:1, 1:4) => :b, (2:3, 1:2) => :d, (2:3, 3:4) => :e, (4:4, 1:4) => :c, (5:6, 1:4) => :f]

  # Avoid key splitting when the set value is identical to the one in the encompassing range.
  smap[Subresource(5:5, 1:4)] = :f
  @test !haskey(smap.value_per_layer, 5:5)
  @test haskey(smap.value_per_layer, 5:6)
  smap[Subresource(5:5, 2:3)] = :f
  @test !haskey(smap.value_per_layer, 5:5)
  @test haskey(smap.value_per_layer, 5:6)
  @test !haskey(smap.value_per_layer[5:6], 2:3)
  @test haskey(smap.value_per_layer[5:6], 1:4)
  validate_map(smap)

  # Limit matched ranges to only what is contained within the requested subresource.
  map = SubresourceMap(6, 4, :a)
  match_subresource(map, Subresource(5:5, 2:3)) do matched_layer_range, matched_mip_range, value
    @test value == :a
    @test matched_layer_range == 5:5
    @test matched_mip_range == 2:3
  end

  smap = SubresourceMap(6, 4, :a)
  smap[Subresource(2:3, 2:4)] = :b
  validate_map(smap)

  @testset "Bounds checking" begin
    smap = SubresourceMap(6, 4, :a)
    @test_throws InvalidLayerRange smap[Subresource(1:7, 1:4)]
    @test_throws InvalidMipRange smap[Subresource(2:3, 1:5)]
    @test_throws InvalidLayerRange query_subresource(smap, Subresource(1:7, 1:4))
    @test_throws InvalidMipRange query_subresource(smap, Subresource(2:3, 1:5))
    @test_throws InvalidMipRange smap[Subresource(2:3, 4:5)] = :b
    @test_throws InvalidLayerRange smap[Subresource(2:8, 1:1)] = :b
  end

  @testset "Robustness check" begin
    subresources = [
      Subresource(1:6, 1:4), # whole range
      Subresource(1:2, 1:2), # partial range
      Subresource(2:6, 2:4), # partial range
      Subresource(2:4, 2:3), # partial range
      Subresource(3:3, 1:4), # partial mip range
      Subresource(3:3, 1:3), # partial mip range
      Subresource(2:3, 1:4), # partial layer range
      Subresource(2:3, 1:1), # partial layer range
      Subresource(3:3, 2:2), # point
    ]
    smap = SubresourceMap(6, 4, :a)
    for (i, subresource) in enumerate([subresources; reverse(subresources)])
      value = Symbol(:value_, i)
      smap[subresource] = value
      @test smap[subresource] == value
      match_subresource(smap, subresource) do matched_layer_range, matched_mip_range, matched_value
        @test matched_value == value
      end
      matched = query_subresource(smap, subresource)
      @test matched == [(subresource.layer_range, subresource.mip_range) => value]
      other = subresources[mod1(7i, length(subresources))]
      matched = query_subresource(smap, other)
      @test !isempty(matched)
    end
  end
end;
