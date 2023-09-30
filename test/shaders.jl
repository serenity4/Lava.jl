function test_shader(position)
  position[] = Vec(1f0, 1f0, 1f0, 1f0)
end

@testset "Shader cache" begin
  @test_throws "`Device` expected" @fragment "hello" test_shader(::Vec4::Output)

  vert_shader = @shader :vertex device test_shader(::Vec4::Output)
  @test isa(vert_shader, Shader)

  frag_shader = @fragment device test_shader(::Vec4::Output)
  @test isa(frag_shader, Shader)

  cache = device.shader_cache
  empty!(cache)
  @test cache.shaders.diagnostics.misses == cache.shaders.diagnostics.hits == 0
  sh = @fragment device test_shader(::Vec4::Output)
  @test cache.shaders.diagnostics.hits == 0
  @test cache.shaders.diagnostics.misses == 1
  sh = @fragment device test_shader(::Vec4::Output)
  @test cache.shaders.diagnostics.hits == 1
  @test cache.shaders.diagnostics.misses == 1
  @test length(cache.shaders) == 1
  sh = @fragment device cached = false test_shader(::Vec4::Output)
  @test cache.shaders.diagnostics.hits == 1
  @test cache.shaders.diagnostics.misses == 1
  @test length(cache.shaders) == 1
  sh = @fragment device test_shader(::Vec{4,Float64}::Output)
  @test cache.shaders.diagnostics.hits == 1
  @test cache.shaders.diagnostics.misses == 2
  @test length(cache.shaders) == 2
end;
