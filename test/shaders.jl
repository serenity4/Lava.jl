instance, device = init(; device_extensions = ["VK_KHR_synchronization2"])
frag_shader = resource("decorations.frag")

@testset "Shader cache" begin
    spec = ShaderSpecification(frag_shader, GLSL)
    cache = device.shader_cache
    Shader(Lava.ShaderCache(device), spec) # trigger JIT compilation
    t = @elapsed Shader(cache, spec)
    @test t > 0.01
    t = @elapsed Shader(cache, spec)
    @test t < 1e-5
end

@testset "Descriptors" begin
    meta = ResourceDescriptors(device)
    @test meta.set ≠ C_NULL
end
