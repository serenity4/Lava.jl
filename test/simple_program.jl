function test_program_vert(position, index, dd)
  pos = Pointer{Vector{Point{2,Float32}}}(dd.vertex_data)[index]
  position[] = Vec(pos[1], pos[2], 0F, 1F)
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

  vert_shader = @shader vert_interface test_program_vert(::Vec{4,Float32}, ::UInt32, ::DrawData)
  frag_shader = @shader frag_interface test_program_frag(::Vec{4,Float32})
  Program(device, vert_shader, frag_shader)
end
