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
  elseif isused(map.value_per_layer)
    @test !isused(map.value_per_mip_level)
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

  smap = SubresourceMap{Symbol}(1, 1)
  @test smap.value == nothing
  sub = Subresource()
  smap[sub] = :a
  @test smap.value == :a
  validate_map(smap)

  smap = SubresourceMap{Symbol}(6, 4)
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
  @test_throws InvalidLayerRange smap[Subresource(1:7, 1:4)]
  @test_throws InvalidMipRange smap[Subresource(2:3, 1:5)]
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
end;
