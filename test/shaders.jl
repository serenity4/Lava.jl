function test_shader(position)
  position[] = Vec(1f0, 1f0, 1f0, 1f0)
end

@testset "Shader cache" begin
  @test_throws "`Device` expected" @eval @fragment "hello" test_shader(::Vec4::Output)
  @test_throws "More than one built-in" @eval @fragment device any_shader(::UInt32::Input{VertexIndex, InstanceIndex})
  @test_throws "Expected macrocall" @eval @fragment device any_shader(::UInt32::Input{VertexIndex, DescriptorSet = 1})
  @test_throws "Unknown storage class" @eval @fragment device any_shader(::UInt32::Typo{VertexIndex, DescriptorSet = 1})
  @test_throws "Unknown decoration" @eval @fragment device any_shader(::UInt32::Input{VertexIndex, @Typo(1)})

  argtypes, scs, vardecs = shader_decorations(:(any_shader(::Vec4::Uniform{@DescriptorSet(1)})))
  @test argtypes == [:Vec4]
  @test scs == [SPIRV.StorageClassUniform]
  @test vardecs == dictionary([1 => Decorations(SPIRV.DecorationDescriptorSet, 1)])

  argtypes, scs, vardecs = shader_decorations(:(any_shader(::UInt32::Input{@Flat})))
  @test argtypes == [:UInt32]
  @test scs == [SPIRV.StorageClassInput]
  @test vardecs == dictionary([1 => Decorations(SPIRV.DecorationFlat).decorate!(SPIRV.DecorationLocation, 0)])

  argtypes, scs, vardecs = shader_decorations(:(any_shader(::UInt32::Input{@Flat}, ::Vec4::Input{Position}, ::Vec4::Input)))
  @test argtypes == [:UInt32, :Vec4, :Vec4]
  @test scs == [SPIRV.StorageClassInput, SPIRV.StorageClassInput, SPIRV.StorageClassInput]
  @test vardecs == dictionary([
    1 => Decorations(SPIRV.DecorationFlat).decorate!(SPIRV.DecorationLocation, 0),
    2 => Decorations(SPIRV.DecorationBuiltIn, SPIRV.BuiltInPosition),
    3 => Decorations(SPIRV.DecorationLocation, 1),
  ])

  frag_shader = @fragment device test_shader(::Vec4::Output)
  @test isa(frag_shader, Shader)

  cache = device.shader_cache
  empty!(cache)
  @test cache.compiled.diagnostics.misses == cache.compiled.diagnostics.hits == 0
  @test cache.shaders.diagnostics.misses == cache.shaders.diagnostics.hits == 0
  sh = @fragment device test_shader(::Vec4::Output)
  @test cache.compiled.diagnostics.hits == 0
  @test cache.compiled.diagnostics.misses == 1
  @test cache.shaders.diagnostics.hits == 0
  @test cache.shaders.diagnostics.misses == 1
  sh = @fragment device test_shader(::Vec4::Output)
  @test cache.compiled.diagnostics.hits == 1
  @test cache.compiled.diagnostics.misses == 1
  @test cache.shaders.diagnostics.hits == 1
  @test cache.shaders.diagnostics.misses == 1
  @test length(cache.compiled) == 1
  @test length(cache.shaders) == 1
  sh = @fragment device test_shader(::Vec{4,Float64}::Output)
  @test cache.compiled.diagnostics.hits == 1
  @test cache.compiled.diagnostics.misses == 2
  @test cache.shaders.diagnostics.hits == 1
  @test cache.shaders.diagnostics.misses == 2
  @test length(cache.compiled) == 2
  @test length(cache.shaders) == 2
end;
