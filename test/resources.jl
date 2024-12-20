@testset "Resources" begin
  resource = Resource(RESOURCE_TYPE_IMAGE, nothing)
  assert_type(resource, RESOURCE_TYPE_IMAGE)
  @test_throws AssertionError assert_type(resource, RESOURCE_TYPE_BUFFER)

  @testset "Logical resources" begin
    buffer = buffer_resource(1024)
    @test islogical(buffer)
    @test resource_type(buffer) == RESOURCE_TYPE_BUFFER
    @test isa(buffer.data, LogicalBuffer)
    @test isa(sprint(show, MIME"text/plain"(), buffer), String)

    image = image_resource(RGBA{Float32}, [16, 16])
    @test islogical(image)
    @test resource_type(image) == RESOURCE_TYPE_IMAGE
    @test isa(image.data, LogicalImage)
    @test isa(sprint(show, MIME"text/plain"(), image), String)

    image_view = image_view_resource(image)
    @test islogical(image_view)
    @test resource_type(image_view) == RESOURCE_TYPE_IMAGE_VIEW
    @test isa(image_view.data, LogicalImageView)
    @test isa(sprint(show, MIME"text/plain"(), image_view), String)

    attachment = attachment_resource(RGBA{Float32}, [16, 16])
    @test islogical(attachment)
    @test resource_type(attachment) == RESOURCE_TYPE_ATTACHMENT
    @test isa(attachment.data, LogicalAttachment)
    @test isa(sprint(show, MIME"text/plain"(), attachment), String)
  end

  @testset "Physical resources" begin
    buffer = buffer_resource(device, ones(UInt8, 1024))
    @test isphysical(buffer)
    @test resource_type(buffer) == RESOURCE_TYPE_BUFFER
    @test isa(buffer.data, Buffer)
    @test isallocated(buffer.data)
    @test isa(sprint(show, MIME"text/plain"(), buffer), String)

    image = image_resource(device, fill(RGBA(0.1f0, 0.1f0, 0.1f0, 1f0), 16, 16))
    @test isphysical(image)
    @test resource_type(image) == RESOURCE_TYPE_IMAGE
    @test isa(image.data, Image)
    @test isallocated(image.data)
    @test isa(sprint(show, MIME"text/plain"(), image), String)

    image_view = image_view_resource(device, fill(RGBA(0.1f0, 0.1f0, 0.1f0, 1f0), 16, 16); mip_levels = 2, mip_range = 2:2)
    @test isphysical(image_view)
    @test resource_type(image_view) == RESOURCE_TYPE_IMAGE_VIEW
    @test isa(image_view.data, ImageView)
    @test isa(sprint(show, MIME"text/plain"(), image_view), String)

    attachment = attachment_resource(device, fill(RGBA(0.1f0, 0.1f0, 0.1f0, 1f0), 16, 16))
    @test isphysical(attachment)
    @test resource_type(attachment) == RESOURCE_TYPE_ATTACHMENT
    @test isa(attachment.data, Attachment)
    @test isallocated(attachment.data.view.image)
    @test isa(sprint(show, MIME"text/plain"(), attachment), String)
  end
end;
