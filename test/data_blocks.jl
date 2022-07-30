using Lava, Test, Dictionaries
using Lava: LogicalDescriptors, patch_descriptors!, patch_pointers!
using SPIRV: TypeInfo, VulkanLayout, align

@testset "Data blocks" begin
  b1 = DataBlock((1, 0x02, 3))
  @test isempty(b1.descriptor_ids)
  @test isempty(b1.pointer_addresses)
  b2 = DataBlock((3U, 0x01, DescriptorIndex(1)))
  @test b2.descriptor_ids == [4 + 1 + 1]
  @test isempty(b2.pointer_addresses)
  b3 = DataBlock((DeviceAddress(b1), DeviceAddress(b2), 3))
  @test isempty(b3.descriptor_ids)
  @test b3.pointer_addresses == [1, 8 + 1]

  layout = VulkanLayout()
  type_info = TypeInfo([b1.type, b2.type, b3.type], layout)
  ab1 = align(b1, type_info)
  @test length(b1.bytes) == 17
  @test length(ab1.bytes) == 24
  ab2 = align(b2, type_info)
  @test length(b2.bytes) == 9
  @test length(ab2.bytes) == 12
  @test ab2.descriptor_ids == [1 + 4 + 4]
  ab3 = align(b3, type_info)
  @test length(b3.bytes) == 24
  @test length(ab3.bytes) == 24
  @test ab3.pointer_addresses == [1, 8 + 1]

  image = Lava.LogicalImage(Lava.uuid(), Vk.FORMAT_UNDEFINED, [1920, 1080], 1, 1)
  tex = Texture(image)
  descriptors = [tex]
  data = ProgramInvocationData([b1, b2, b3], descriptors, 3)
  @test data.postorder_traversal == [1, 2, 3]

  ldescs = LogicalDescriptors()
  @test Base.unsafe_load(Ptr{UInt32}(pointer(@view b2.bytes[only(b2.descriptor_ids)]))) == 1U
  patch_descriptors!(b2, ldescs, data.descriptors, Lava.uuid())
  @test Base.unsafe_load(Ptr{UInt32}(pointer(@view b2.bytes[only(b2.descriptor_ids)]))) == 0U

  addresses = Dictionary(objectid.([b1, b2]), UInt64[1, 2])
  patch_pointers!(b3, addresses)
  patched_addresses = [Base.unsafe_load(Ptr{UInt64}(pointer(@view b3.bytes[address_byte]))) for address_byte in b3.pointer_addresses]
  @test patched_addresses == [1, 2]

  # TODO: Test `device_address_block!`.
  # allocator = LinearAllocator(device, 1_000)

  # TODO: Define macro.
  #=
  data = @invocation_data begin
    b1 = @block (1, 0x02, 3)
    b2 = @block (3U, 0x01, @descriptor(tex))
    # The last block index will be set as root.
    b3 = @block (@address(b1), @address(b2), 3)
  end
  @test data.root == 3
  @test data.blocks[1].bytes == b1.bytes
  @test data.blocks[2].bytes == b2.bytes
  @test data.blocks[2].descriptor_ids == b2.descriptor_ids
  @test data.blocks[3].bytes == b3.bytes
  @test length(data.blocks[3].pointer_addresses) == length(b3.pointer_addresses)
  =#
end
