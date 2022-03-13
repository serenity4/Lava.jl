using Base: UUID

@testset "Resources" begin
  resources = Resources()
  b = UUID(new!(resources, LogicalBuffer(1024)))
  @test isnothing(Lava.resource_data(resources, b))
  @test haskey(resources, b)
  delete!(resources, b)
  @test !haskey(resources, b)
  @test_throws IndexError Lava.resource_data(resources, b)

  data = buffer(device, 1024)
  b2 = UUID(new!(resources, data))
  @test b ≠ b2
  @test Lava.resource_data(resources, b2) === data
  @test haskey(resources, b2)
  delete!(resources, b2)
  @test !haskey(resources, b2)

  img = UUID(new!(resources, LogicalImage(Vk.FORMAT_R32G32B32A32_SFLOAT)))
  @test isnothing(Lava.resource_data(resources, img))
  delete!(resources, img)

  data = image(device, rand(RGBA{Float32}, 64, 64), Vk.FORMAT_R32G32B32A32_SFLOAT)
  img2 = UUID(new!(resources, data))
  @test img ≠ img2
  @test Lava.resource_data(resources, img2) === data
  delete!(resources, img2, false)

  img3 = UUID(new!(device, data))
  @test haskey(device.resources, img3)
  delete!(device, img3)
  @test !haskey(device.resources, img3)

  @test haskey(resources, UUID(buffer_resource!(resources, 1024)))
  @test haskey(resources, UUID(image_resource!(resources, Vk.FORMAT_R32G32B32A32_SFLOAT)))
  @test haskey(resources, UUID(attachment_resource!(resources, Vk.FORMAT_R32G32B32A32_SFLOAT)))
end
