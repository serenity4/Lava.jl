function test_program_vert(position, index, data_address::DeviceAddressBlock)
  pos = @load data_address[index]::Vec2
  position[] = Vec(pos.x, pos.y, 0F, 1F)
end

function test_program_frag(out_color)
  out_color[] = Vec(1F, 0F, 0F, 0F)
end

function simple_program(device)
  vert_shader = @vertex device test_program_vert(::Vec4::Output{Position}, ::UInt32::Input{VertexIndex}, ::DeviceAddressBlock::PushConstant)
  frag_shader = @fragment device test_program_frag(::Vec4::Output)
  Program(vert_shader, frag_shader)
end
