@testset "Resources" begin
  r = Resource(RESOURCE_TYPE_IMAGE, nothing)
  assert_type(r, RESOURCE_TYPE_IMAGE)
  @test_throws AssertionError assert_type(r, RESOURCE_TYPE_BUFFER)

  @testset "Logical resources" begin
    b = buffer_resource(1024)
    @test islogical(b)
    @test resource_type(b) == RESOURCE_TYPE_BUFFER
    @test isa(b.data, LogicalBuffer)

    im = image_resource(RGBA{Float32}, [16, 16])
    @test islogical(im)
    @test resource_type(im) == RESOURCE_TYPE_IMAGE
    @test isa(im.data, LogicalImage)

    att = attachment_resource(RGBA{Float32}, [16, 16])
    @test islogical(att)
    @test resource_type(att) == RESOURCE_TYPE_ATTACHMENT
    @test isa(att.data, LogicalAttachment)
  end

  @testset "Physical resources" begin
    b = buffer_resource(device, ones(UInt8, 1024))
    @test isphysical(b)
    @test resource_type(b) == RESOURCE_TYPE_BUFFER
    @test isa(b.data, Buffer)
    @test isallocated(b.data)

    im = image_resource(device, fill(RGBA(0.1f0, 0.1f0, 0.1f0, 1f0), 16, 16))
    @test isphysical(im)
    @test resource_type(im) == RESOURCE_TYPE_IMAGE
    @test isa(im.data, Image)
    @test isallocated(im.data)

    att = attachment_resource(device, fill(RGBA(0.1f0, 0.1f0, 0.1f0, 1f0), 16, 16))
    @test isphysical(att)
    @test resource_type(att) == RESOURCE_TYPE_ATTACHMENT
    @test isa(att.data, Attachment)
    @test isallocated(att.data.view.image)
  end
end;
