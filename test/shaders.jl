frag_shader = resource("dummy.frag")

@testset "Shader cache" begin
    spec = ShaderSpecification(frag_shader, GLSL)
    cache = Lava.ShaderCache(device)
    Shader(Lava.ShaderCache(device), spec) # trigger JIT compilation
    t = @elapsed Shader(cache, spec)
    @test t > 0.01
    t = @elapsed Shader(cache, spec)
    @test t < 1e-5
end
