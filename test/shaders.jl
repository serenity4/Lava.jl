instance, device = init(; device_extensions = ["VK_KHR_synchronization2"])
frag_shader = resource("decorations.frag")

@testset "Shader cache" begin
    spec = ShaderSpecification(frag_shader, GLSL)
    cache = Lava.ShaderCache(device)
    Lava.find_shader!(Lava.ShaderCache(device), spec) # trigger JIT compilation
    t = @elapsed Lava.find_shader!(cache, spec)
    @test t > 0.01
    t = @elapsed Lava.find_shader!(cache, spec)
    @test t < 1e-5
end

@testset "Descriptors" begin
    spec = ShaderSpecification(frag_shader, GLSL)
    cache = Lava.ShaderCache(device)
    shader = Lava.find_shader!(cache, spec)
end
