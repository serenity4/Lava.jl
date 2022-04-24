@testset "Resources" begin
  @testset "Logical resources" begin
    resources = Lava.LogicalResources()
    b = buffer(resources, 1024)
    @test b isa LogicalBuffer
    @test resources[b.uuid] === b

    im = image(resources, Vk.FORMAT_R32G32B32A32_SFLOAT, (16, 16))
    @test im isa LogicalImage
    @test resources[im.uuid] === im

    att = attachment(resources, Vk.FORMAT_R32G32B32A32_SFLOAT, (16, 16))
    @test att isa LogicalAttachment
    @test resources[att.uuid] === att
  end

  @testset "Physical resources" begin
    resources = Lava.PhysicalResources()
    b = buffer(device; size = 1024)
    r = buffer(resources, b)
    @test r isa PhysicalBuffer
    @test resources[r.uuid] === r

    im = image(device; format = Vk.FORMAT_R32G32B32A32_SFLOAT, dims = (16, 16))
    r = image(resources, im)
    @test r isa PhysicalImage
    @test resources[r.uuid] === r

    att = attachment(device; format = Vk.FORMAT_R32G32B32A32_SFLOAT, dims = (16, 16))
    r = attachment(resources, att)
    @test r isa PhysicalAttachment
    @test resources[r.uuid] === r
  end
end
