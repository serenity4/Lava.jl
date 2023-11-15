@testset "Descriptors" begin
  image = image_resource(Vk.FORMAT_UNDEFINED, [1920, 1080])
  sampling = Sampling()
  texture = Texture(image, sampling)
  d1 = storage_image_descriptor(image)
  d2 = sampler_descriptor(sampling)
  d3 = sampled_image_descriptor(image)
  d4 = texture_descriptor(texture)
  @test all(isa(d, Descriptor) for d in [d1, d2, d3, d4])

  @testset "Descriptor arrays" begin
    arr = DescriptorArray()
    id = DescriptorID(DESCRIPTOR_TYPE_TEXTURE)
    index = new_descriptor!(arr, id)
    @test index == 1
    index = new_descriptor!(arr, id)
    @test index == 1
    id2 = DescriptorID(DESCRIPTOR_TYPE_TEXTURE)
    index = new_descriptor!(arr, id2)
    @test index == 2
    delete_descriptor!(arr, id)
    @test 1 in arr.holes
    id3 = DescriptorID(DESCRIPTOR_TYPE_TEXTURE)
    index = new_descriptor!(arr, id3)
    @test index == 1
    @test length(arr.holes) == 0
    @test_throws Dictionaries.IndexError delete_descriptor!(arr, id)
    delete_descriptor!(arr, id2)
    delete_descriptor!(arr, id3)
    @test isempty(arr.descriptors)
  end

  @testset "Global descriptors" begin
    gdescs = GlobalDescriptors(device)

    idx = request_index!(gdescs, texture_descriptor(texture))
    @test idx == 1
    @test length(gdescs.descriptors) == 1

    idx = request_index!(gdescs, d4)
    @test idx == 2
    @test length(gdescs.descriptors) == 2
    insert!(gdescs.pending, 1, [d4.id])

    free_descriptor_batch!(gdescs, 1)
    @test length(gdescs.descriptors) == 1
    @test !haskey(gdescs.descriptors, d4.id)

    empty!(gdescs)
    @test isempty(gdescs.descriptors)
  end
end;
