using Lava, Test, Dictionaries
using Lava: generated_block_address, generated_logical_buffer_address, requires_annotations
using BenchmarkTools: @btime

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

data = [Arr{3,Vec2}(rand(Vec2, 3)) for _ in 1:100]
ctx = InvocationDataContext(layout)
@btime DataBlock($data, $ctx)

@profview for _ in 1:100000; DataBlock(data, ctx); end
@descend DataBlock(data, ctx)
