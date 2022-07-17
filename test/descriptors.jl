@testset "Descriptors" begin
  @testset "Descriptor arrays" begin
    arr = Lava.DescriptorArray()
    id = Lava.uuid()
    index = Lava.new_descriptor!(arr, id)
    @test index == 0
    index = Lava.new_descriptor!(arr, id)
    @test index == 0
    id2 = Lava.uuid()
    index = Lava.new_descriptor!(arr, id2)
    @test index == 1
    Lava.delete_descriptor!(arr, id)
    @test 0 in arr.holes
    id3 = Lava.uuid()
    index = Lava.new_descriptor!(arr, id3)
    @test index == 0
    @test length(arr.holes) == 0
    @test_throws Dictionaries.IndexError Lava.delete_descriptor!(arr, id)
    Lava.delete_descriptor!(arr, id2)
    Lava.delete_descriptor!(arr, id3)
    @test isempty(arr.descriptors)
  end

  @testset "Logical descriptors" begin
    ldescs = Lava.LogicalDescriptors()
    id = Lava.uuid()
    image = Lava.LogicalImage(id, Vk.FORMAT_UNDEFINED, [1920, 1080], 1, 1)
    tex = Texture(image)
    node_id = Lava.uuid()
    idx = request_descriptor_index(ldescs, node_id, tex)
    @test idx == 0
    @test length(ldescs.textures) == 1
  end
end;
