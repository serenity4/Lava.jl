function test_program_vert(position, index, data_address::DeviceAddressBlock)
  pos = Pointer{Vector{Vec2}}(data_address)[index]
  position[] = Vec(pos.x, pos.y, 0F, 1F)
end

function test_program_frag(out_color)
  out_color[] = Vec(1F, 0F, 0F, 0F)
end

function simple_program(device)
  vert_interface = ShaderInterface(
    storage_classes = [SPIRV.StorageClassOutput, SPIRV.StorageClassInput, SPIRV.StorageClassPushConstant],
    variable_decorations = dictionary([
      1 => Decorations(SPIRV.DecorationBuiltIn, SPIRV.BuiltInPosition),
      2 => Decorations(SPIRV.DecorationBuiltIn, SPIRV.BuiltInVertexIndex),
    ]),
    features = device.spirv_features,
  )

  frag_interface = ShaderInterface(
    execution_model = SPIRV.ExecutionModelFragment,
    storage_classes = [SPIRV.StorageClassOutput],
    variable_decorations = dictionary([
      1 => Decorations(SPIRV.DecorationLocation, 0),
    ]),
    features = device.spirv_features,
  )

  vert_shader = @shader vert_interface test_program_vert(::Vec4, ::UInt32, ::DeviceAddressBlock)
  frag_shader = @shader frag_interface test_program_frag(::Vec4)
  Program(device, vert_shader, frag_shader)
end
