function test_shader(position)
  position[] = Vec(1f0, 1f0, 1f0, 1f0)
end

@testset "Shader cache" begin
  frag_shader = @fragment device.spirv_features test_shader(::Output::Vec{4,Float32})

  cache = Lava.ShaderCache(device)
  Shader(Lava.ShaderCache(device), frag_shader) # trigger JIT compilation
  t1 = @elapsed Shader(cache, frag_shader)
  t2 = @elapsed Shader(cache, frag_shader)
  @test length(cache.shaders) == 1
  @test t1 > t2

  frag_shader = @fragment device.spirv_features test_shader(::Output::Vec{4,Float32})
  t3 = @elapsed Shader(cache, frag_shader)
  @test length(cache.shaders) == 1
  @test t1 > t3
end
