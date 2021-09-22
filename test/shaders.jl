instance, device = init(; device_extensions = ["VK_KHR_synchronization2"])
frag_shader = resource("decorations.frag")

@testset "Shader cache" begin
    spec = ShaderSpecification(frag_shader, GLSL)
    cache = device.shader_cache
    Lava.find_shader!(Lava.ShaderCache(device), spec) # trigger JIT compilation
    t = @elapsed Lava.find_shader!(cache, spec)
    @test t > 0.01
    t = @elapsed Lava.find_shader!(cache, spec)
    @test t < 1e-5
end

@testset "Descriptors" begin
    spec = ShaderSpecification(frag_shader, GLSL)
    cache = device.shader_cache
    shader = Lava.find_shader!(cache, spec)
    set_layout = first(create_descriptor_set_layouts([shader]))
    da = DescriptorAllocator(set_layout)
    allocate_pool(da)
    pool = first(da.pools)
    @test pool.allocated == 0
    set = allocate_descriptor_set(da)
    @test set.layout === set_layout
    @test pool.allocated == 1
    da = DescriptorAllocator(set_layout)
    @test allocate_descriptor_set(da) isa DescriptorSet
end
