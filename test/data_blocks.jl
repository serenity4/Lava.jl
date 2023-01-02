using Lava, Test, Dictionaries

using SPIRV: TypeInfo, VulkanLayout, align

layout = VulkanLayout()

function data_blocks()
  b1 = DataBlock((1, 0x02, 3))
  b2 = DataBlock((3U, 0x01, DescriptorIndex(1)))
  b3 = DataBlock((DeviceAddress(b1), DeviceAddress(b2), 3))
  [b1, b2, b3]
end

function data_blocks_2()
  b1 = DataBlock([(1, 2), (2, 3)])
  b2 = DataBlock([(DeviceAddress(b1), DescriptorIndex(1))])
  [b1, b2]
end

function data_blocks_3()
  b1 = DataBlock([
    (Vec2(-0.5, 0.5), Arr{Float32}(1.0, 0.0, 0.0)),
    (Vec2(-0.5, -0.5), Arr{Float32}(0.0, 1.0, 0.0)),
    (Vec2(0.5, 0.5), Arr{Float32}(1.0, 1.0, 1.0)),
    (Vec2(0.5, -0.5), Arr{Float32}(0.0, 0.0, 1.0)),
  ])
  b2 = DataBlock((Vec2(0.1, 1.0), DescriptorIndex(1)))
  b3 = DataBlock((DeviceAddress(b1), DeviceAddress(b2)))
  [b1, b2, b3]
end

infer_type_info(blocks) = TypeInfo(getproperty.(blocks, :type), layout)
infer_type_info(block::DataBlock) = infer_type_info([block])
infer_type_info(data::ProgramInvocationData) = infer_type_info(data.blocks)

type_info = infer_type_info([data_blocks(); data_blocks_2(); data_blocks_3()])

function test_align_block(b::DataBlock, type_info::TypeInfo = type_info)
  aligned = align(b, type_info)
  @test length(aligned.descriptor_ids) == length(b.descriptor_ids)
  @test length(aligned.pointer_addresses) == length(b.pointer_addresses)
  @test length(aligned.bytes) ≥ length(b.bytes)
  @test aligned.type === b.type
end

@testset "Data blocks" begin
  b1, b2, b3 = data_blocks()
  @test isempty(b1.descriptor_ids)
  @test isempty(b1.pointer_addresses)
  @test b2.descriptor_ids == [1 + 4 + 1]
  @test isempty(b2.pointer_addresses)
  b3 = DataBlock((DeviceAddress(b1), DeviceAddress(b2), 3))
  @test isempty(b3.descriptor_ids)
  @test b3.pointer_addresses == [1, 8 + 1]

  foreach(test_align_block, data_blocks())
  foreach(test_align_block, data_blocks_2())
  foreach(test_align_block, data_blocks_3())

  b1, b2, b3 = data_blocks()
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

  b1, b2, b3 = data_blocks_3()
  ab1 = align(b1, type_info)
  @test length(ab1.bytes) == (24 * 3) + 20
  ab2 = align(b2, type_info)
  ab3 = align(b3, type_info)
  ab3.pointer_addresses
end

# `tex` is put in global scope to test for the hygiene of `@invocation_data`.
img = image_resource(Vk.FORMAT_UNDEFINED, [1920, 1080])
tex = Texture(img)
desc = texture_descriptor(tex)

pointer_addresses(block::DataBlock) = [Base.unsafe_load(Ptr{UInt64}(pointer(@view block.bytes[address_byte]))) for address_byte in block.pointer_addresses]

@testset "Program invocation data & block transforms" begin
  b1, b2, b3 = data_blocks()
  descriptors = [desc]
  foreach(test_align_block, [b1, b2, b3])

  gdescs = GlobalDescriptors(device)
  @test Base.unsafe_load(Ptr{UInt32}(pointer(@view b2.bytes[only(b2.descriptor_ids)]))) == 1U
  patch_descriptors!(b2, gdescs, descriptors, NodeID())
  @test Base.unsafe_load(Ptr{UInt32}(pointer(@view b2.bytes[only(b2.descriptor_ids)]))) == 0U

  addresses = Dictionary(objectid.([b1, b2]), UInt64[1, 2])
  patch_pointers!(b3, addresses)
  @test pointer_addresses(b3) == [1, 2]
  @test_throws "Bad pointer dependency order" patch_pointers!(b3, addresses)
  @test pointer_addresses(b3) == [1, 2]

  allocator = LinearAllocator(device, 1_000)
  data = ProgramInvocationData(data_blocks(), descriptors, 3)
  @test data.postorder_traversal == [1, 2, 3]
  @test_throws "different descriptor" device_address_block!(allocator, gdescs, NodeID(), data, type_info, layout)
  empty!(gdescs)
  address = device_address_block!(allocator, gdescs, NodeID(), data, type_info, layout)
  @test isa(address, DeviceAddressBlock)

  b1, b2, b3 = data_blocks_3()
  addresses = Dictionary(objectid.([b1, b2]), UInt64[1, 2])
  patch_pointers!(b3, addresses)
  @test pointer_addresses(b3) == [1, 2]

  data2 = @invocation_data begin
    b1 = @block (1, 0x02, 3)
    b2 = @block (3U, 0x01, @descriptor(desc))
    # The last block index will be set as root.
    b3 = @block (@address(b1), @address(b2), 3)
  end
  @test data2.root == data.root
  @test data2.blocks[1].bytes == data.blocks[1].bytes
  @test data2.blocks[2].bytes == data.blocks[2].bytes
  @test data2.blocks[2].descriptor_ids == data.blocks[2].descriptor_ids
  @test length(data2.blocks[3].bytes) == length(data.blocks[3].bytes)
  @test length(data.blocks[3].pointer_addresses) == length(data.blocks[3].pointer_addresses)

  # Make sure we got the hygiene right.
  M = Module()
  ex = macroexpand(M, :($(@__MODULE__).@invocation_data begin
      b1 = @block (1, 0x02, 3)
      b2 = @block (3 * $(@__MODULE__).U, 0x01, @descriptor($(@__MODULE__).desc))
      # The last block index will be set as root.
      b3 = @block (@address(b1), @address(b2), 3)
    end))
  @test isa(Core.eval(M, ex), ProgramInvocationData)

  # Support for array blocks.
  allocator = LinearAllocator(device, 1_000)
  data3 = @invocation_data begin
    b1 = @block [(1, 2), (2, 3)]
    @block [(@address(b1), @descriptor(desc))]
  end
  @test data3.postorder_traversal == [1, 2]
  type_info2 = TypeInfo(getproperty.(data3.blocks, :type), layout)
  empty!(gdescs)
  address = device_address_block!(allocator, gdescs, NodeID(), data3, type_info2, layout)
  @test isa(address, DeviceAddressBlock)

  data4 = @invocation_data begin
    b1 = @block [
      (Vec2(-0.5, 0.5), Arr{Float32}(1.0, 0.0, 0.0)),
      (Vec2(-0.5, -0.5), Arr{Float32}(0.0, 1.0, 0.0)),
      (Vec2(0.5, 0.5), Arr{Float32}(1.0, 1.0, 1.0)),
      (Vec2(0.5, -0.5), Arr{Float32}(0.0, 0.0, 1.0)),
    ]
    b2 = @block((Vec2(0.1, 1.0), @descriptor desc))
    @block ((@address(b1), @address(b2)))
  end
  @test data4.blocks[3].pointer_addresses == [1, 9]
  @test data4.blocks[2].descriptor_ids == [9]
  @test align(data4.blocks[3], type_info).pointer_addresses == [1, 9]
  @test align(data4.blocks[2], type_info).descriptor_ids == [9]

  data5 = @invocation_data begin
    b1 = @block Vec2(1.0, 1.0)
    b2 = @block (Vec2(1, 1), @descriptor desc)
    tex = (@address(b1), @address(b2))
    @block (tex, 4)
  end
  empty!(gdescs)
  address = device_address_block!(allocator, gdescs, NodeID(), data5, infer_type_info(data5), layout)
  @test isa(address, DeviceAddressBlock)
end;
