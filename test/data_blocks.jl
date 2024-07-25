using Lava, Test, Dictionaries
using Lava: generated_block_address, generated_logical_buffer_address

using SPIRV: VulkanLayout

layout = VulkanLayout([
  Tuple{Int64, UInt8, Int64},
  Tuple{UInt32, UInt8, DescriptorIndex},
  Tuple{DeviceAddress, DeviceAddress, Int64},
  Tuple{DeviceAddress, DeviceAddress},
  Vector{Tuple{Int64, Int64}},
  Vector{DeviceAddress},
  Tuple{Vec2, Arr{3, Float32}},
  Tuple{Vec2, DescriptorIndex},
  Tuple{Vec2, DeviceAddress, DescriptorIndex},
  Tuple{DeviceAddress, DescriptorIndex},
  Tuple{Tuple{DeviceAddress, DeviceAddress}, Int64},
  Tuple{Int64, UInt32},
  Tuple{Int64, DeviceAddress},
])

function data_blocks()
  b1 = DataBlock((1, 0x02, 3), layout)
  b2 = DataBlock((3U, 0x01, DescriptorIndex(1)), layout)
  b3 = DataBlock((generated_block_address(1), generated_block_address(2), 3), layout)
  [b1, b2, b3]
end

function data_blocks_2()
  b1 = DataBlock([
    (Vec2(-0.5, 0.5), @arr Float32[3.0, 5.0, 1.0, 0.0, 0.0]),
    (Vec2(-0.5, -0.5), @arr Float32[3.0, 5.0, 0.0, 1.0, 0.0]),
    (Vec2(0.5, 0.5), @arr Float32[3.0, 5.0, 1.0, 1.0, 1.0]),
    (Vec2(0.5, -0.5), @arr Float32[3.0, 5.0, 0.0, 0.0, 1.0]),
  ], layout)
  b2 = DataBlock((Vec2(0.1, 1.0), DescriptorIndex(1)), layout)
  b3 = DataBlock((generated_block_address(1), generated_block_address(2)), layout)
  [b1, b2, b3]
end

function data_blocks_3()
  b1 = DataBlock([
    (Vec2(-0.5, 0.5), @arr Float32[3.0, 5.0, 1.0, 0.0, 0.0]),
    (Vec2(-0.5, -0.5), @arr Float32[3.0, 5.0, 0.0, 1.0, 0.0]),
    (Vec2(0.5, 0.5), @arr Float32[3.0, 5.0, 1.0, 1.0, 1.0]),
    (Vec2(0.5, -0.5), @arr Float32[3.0, 5.0, 0.0, 0.0, 1.0]),
  ], layout)
  b2 = DataBlock((Vec2(0.1, 1.0), generated_logical_buffer_address(1), DescriptorIndex(1)), layout)
  b3 = DataBlock((generated_block_address(1), generated_block_address(2)), layout)
  [b1, b2, b3]
end

@testset "Data blocks" begin
  b1, b2, b3 = data_blocks()
  @test isempty(b1.descriptor_ids)
  @test isempty(b1.device_addresses)
  @test b2.descriptor_ids == [9]
  @test isempty(b2.device_addresses)
  b3 = DataBlock((generated_block_address(1), generated_block_address(2), 3), layout)
  @test isempty(b3.descriptor_ids)
  @test b3.device_addresses == [1, 9]
  b4 = DataBlock([DescriptorIndex(1), DescriptorIndex(2)], layout)
  @test b4.descriptor_ids == [1, 5]
end

# `tex` is put in global scope to test for the hygiene of `@invocation_data`.
img = image_resource(Vk.FORMAT_UNDEFINED, [1920, 1080])
tex = Texture(img)
desc = texture_descriptor(tex)

buffer = buffer_resource(512)
pbuffer = buffer_resource(device, 512)
@reset pbuffer.id = buffer.id
buffers = dictionary([pbuffer.id => pbuffer])

device_addresses(block::DataBlock) = [Base.unsafe_load(Ptr{UInt64}(pointer(@view block.bytes[address_byte]))) for address_byte in block.device_addresses]

@testset "Program invocation data & block transforms" begin
  b1, b2, b3 = data_blocks()
  descriptors = [desc]

  gdescs = GlobalDescriptors(device)
  @test Base.unsafe_load(Ptr{UInt32}(pointer(@view b2.bytes[only(b2.descriptor_ids)]))) == 1U
  patch_descriptors!(b2, gdescs, descriptors, NodeID())
  @test Base.unsafe_load(Ptr{UInt32}(pointer(@view b2.bytes[only(b2.descriptor_ids)]))) == 1U

  data = ProgramInvocationData([b1, b2, b3], descriptors, [], 3, layout)
  addresses = Dictionary([b1, b2], DeviceAddress[5, 6])
  patch_pointers!(b3, data, addresses, nothing)
  @test device_addresses(b3) == [5, 6]
  @test_throws "Bad pointer dependency order" patch_pointers!(last(data_blocks()), data, empty!(addresses), nothing)
  @test device_addresses(b3) == [5, 6]

  b1, b2, b3 = data_blocks_2()
  data = ProgramInvocationData([b1, b2, b3], descriptors, [], 3, layout)
  addresses = Dictionary([b1, b2], DeviceAddress[5, 6])
  patch_pointers!(b3, data, addresses, nothing)
  @test device_addresses(b3) == [5, 6]

  b1, b2, b3 = data_blocks_3()
  data = ProgramInvocationData([b1, b2, b3], descriptors, [buffer.id], 3, layout)
  addresses = Dictionary([b1, b2], DeviceAddress[5, 6])
  patch_pointers!(b2, data, addresses, buffers)
  @test device_addresses(b2) == UInt64[DeviceAddress(pbuffer)]
  patch_pointers!(b3, data, addresses, buffers)
  @test device_addresses(b3) == [5, 6]

  allocator = LinearAllocator(device, 1_000)
  data = ProgramInvocationData(data_blocks(), descriptors, [], 3, layout)
  @test data.postorder_traversal == [1, 2, 3]
  @test_throws "different descriptor" device_address_block!(allocator, gdescs, nothing, NodeID(), data)
  empty!(gdescs)
  address = device_address_block!(allocator, gdescs, nothing, NodeID(), data)
  @test isa(address, DeviceAddressBlock)

  fake_program = Program(Lava.PROGRAM_TYPE_GRAPHICS, nothing, layout)

  data2 = @invocation_data fake_program begin
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
  @test length(data.blocks[3].device_addresses) == length(data.blocks[3].device_addresses)

  # Make sure we got the hygiene right.
  M = Module()
  Core.eval(M, :(fake_program = $fake_program))
  ex = macroexpand(M, :($(@__MODULE__).@invocation_data fake_program begin
      b1 = @block (1, 0x02, 3)
      b2 = @block (3 * $(@__MODULE__).U, 0x01, @descriptor($(@__MODULE__).desc))
      # The last block index will be set as root.
      b3 = @block (@address(b1), @address(b2), 3)
    end))
  @test isa(Core.eval(M, ex), ProgramInvocationData)

  # Support for array blocks.
  allocator = LinearAllocator(device, 1_000)
  data3 = @invocation_data fake_program begin
    b1 = @block [(1, 2), (2, 3)]
    @block [(@address(b1), @descriptor(desc))]
  end
  @test data3.postorder_traversal == [1, 2]
  empty!(gdescs)
  address = device_address_block!(allocator, gdescs, nothing, NodeID(), data3)
  @test isa(address, DeviceAddressBlock)

  data4 = @invocation_data fake_program begin
    b1 = @block [
      (Vec2(-0.5, 0.5), @arr Float32[3.0, 5.0, 1.0, 0.0, 0.0]),
      (Vec2(-0.5, -0.5), @arr Float32[3.0, 5.0, 0.0, 1.0, 0.0]),
      (Vec2(0.5, 0.5), @arr Float32[3.0, 5.0, 1.0, 1.0, 1.0]),
      (Vec2(0.5, -0.5), @arr Float32[3.0, 5.0, 0.0, 0.0, 1.0]),
    ]
    b2 = @block((Vec2(0.1, 1.0), @descriptor desc))
    @block ((@address(b1), @address(b2)))
  end
  @test data4.blocks[3].device_addresses == [1, 9]
  @test data4.blocks[2].descriptor_ids == [9]

  data5 = @invocation_data fake_program begin
    b1 = @block Vec2(1.0, 1.0)
    b2 = @block (Vec2(1, 1), @descriptor desc)
    tex = (@address(b1), @address(b2))
    @block (tex, 4)
  end
  empty!(gdescs)
  address = device_address_block!(allocator, gdescs, nothing, NodeID(), data5)
  @test isa(address, DeviceAddressBlock)

  data6 = @invocation_data fake_program begin
    b1 = @block (1, 2U)
    @block (@address(buffer), @address(b1))
  end
  address = device_address_block!(allocator, gdescs, buffers, NodeID(), data6)
  @test isa(address, DeviceAddressBlock)

  # Support for descriptor indices/device addresses nested in arrays.
  data7 = @invocation_data fake_program begin
    b1 = @block [@descriptor(desc), @descriptor(desc)]
    b2 = @block (2, @address b1)
    b3 = @block (3, @address b2)
    @block [@address(b2), @address(b3)]
  end
  @test data7.descriptors == [desc]
  @test length(data7.blocks) == 4
  b1, b2, b3, b4 = data7.blocks
  @test b4.descriptor_ids == b3.descriptor_ids == b2.descriptor_ids == []
  @test b4.device_addresses == [1, 9]
  @test b3.device_addresses == b2.device_addresses == [9]
  @test b1.device_addresses == []
  @test b1.descriptor_ids == [1, 5]
  empty!(gdescs)
  address = device_address_block!(allocator, gdescs, buffers, NodeID(), data7)
  @test isa(address, DeviceAddressBlock)
  descs = collect(gdescs.descriptors)
  @test length(descs) == 1
  @test descs[1] == @set(desc.node_id = descs[1].node_id)
end;
