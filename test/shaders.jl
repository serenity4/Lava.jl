function test_frag_shader(out_color, frag_color)
  out_color[] = frag_color
end

@testset "Shader cache" begin
  frag_interface = ShaderInterface(
    execution_model = SPIRV.ExecutionModelFragment,
    storage_classes = [SPIRV.StorageClassOutput, SPIRV.StorageClassInput],
    variable_decorations = dictionary([
      1 => dictionary([SPIRV.DecorationLocation => UInt32[0]]),
      2 => dictionary([SPIRV.DecorationLocation => UInt32[0]]),
    ]),
    features = SPIRV_FEATURES,
  )
  frag_shader = @shader frag_interface test_frag_shader(::Vec{4,Float32}, ::Vec{4,Float32})

  cache = Lava.ShaderCache(device)
  Shader(Lava.ShaderCache(device), frag_shader) # trigger JIT compilation
  t1 = @elapsed Shader(cache, frag_shader)
  t2 = @elapsed Shader(cache, frag_shader)
  @test t1 > t2
end
