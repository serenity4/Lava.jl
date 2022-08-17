function test_shader(position)
  position[] = Vec(1f0, 1f0, 1f0, 1f0)
end

@testset "Shader cache" begin
  frag_shader = @fragment device.spirv_features test_shader(::Output::Vec{4,Float32})

  cache = ShaderCache(device)
  @test cache.diagnostics.misses == cache.diagnostics.hits == 0
  Shader(ShaderCache(device), frag_shader) # trigger JIT compilation
  t1 = @elapsed Shader(cache, frag_shader)
  @test cache.diagnostics.hits == 0
  @test cache.diagnostics.misses == 1
  t2 = @elapsed Shader(cache, frag_shader)
  @test cache.diagnostics.hits == 1
  @test cache.diagnostics.misses == 1
  @test length(cache.shaders) == 1

  # Test that caching is based on equality and not identity.
  # Note that this means caching is slow at the moment since it
  # has to hash the whole source code. An appropriate caching by
  # object ID could be implemented as an optimization option in the future.
  frag_shader = @fragment device.spirv_features test_shader(::Output::Vec{4,Float32})
  t3 = @elapsed Shader(cache, frag_shader)
  @test length(cache.shaders) == 1
  @test cache.diagnostics.hits == 2
end;
