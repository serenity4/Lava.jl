struct TextureDrawing
  uv_scaling::Vec{2,Float32}
  img_index::DescriptorIndex
end

struct TextureData
  coords::DeviceAddress
  drawing::DeviceAddress
end

function texture_vert(uv, position, index, data_address::DeviceAddressBlock)
  data = @load data_address::TextureData
  coords = @load data.coords[index]::TextureCoordinates
  (; pos) = coords
  position[] = Vec(pos.x, pos.y, 0F, 1F)
  uv[] = coords.uv
end

function texture_frag(out_color, uv, data_address, images)
  data = @load data_address::TextureData
  drawing = @load data.drawing::TextureDrawing
  (; uv_scaling, img_index) = drawing
  texcolor = images[img_index](uv * uv_scaling)
  out_color[] = Vec(texcolor.r, texcolor.g, texcolor.b, 1F)
end

function texture_program(device)
  vert = @vertex device.spirv_features texture_vert(::Output::Vec2, ::Output{Position}::Vec4, ::Input{VertexIndex}::UInt32, ::PushConstant::DeviceAddressBlock)
  frag = @fragment device.spirv_features texture_frag(
    ::Output::Vec4,
    ::Input::Vec2,
    ::PushConstant::DeviceAddressBlock,
    ::UniformConstant{DescriptorSet = 0, Binding = 3}::Arr{2048,SPIRV.SampledImage{SPIRV.image_type(SPIRV.ImageFormatRgba16f, SPIRV.Dim2D, 0, false, false, 1)}})
  Program(device, vert, frag)
end

function texture_invocation(device, vdata, color; prog = texture_program(device), normal_map = nothing)
  normal_map = @something(normal_map, read_normal_map(device))
  normal_map_texture = texture_descriptor(Texture(normal_map, setproperties(DEFAULT_SAMPLING, (magnification = Vk.FILTER_LINEAR, minification = Vk.FILTER_LINEAR))))
  invocation_data = @invocation_data begin
    b1 = @block vdata
    b2 = @block TextureDrawing(Vec2(0.1, 1.0), @descriptor normal_map_texture)
    @block TextureData(@address(b1), @address(b2))
  end
  ProgramInvocation(
    prog,
    DrawIndexed(1:4),
    RenderTargets(color),
    invocation_data,
    RenderState(),
    setproperties(ProgramInvocationState(), (;
      primitive_topology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
      triangle_orientation = Vk.FRONT_FACE_COUNTER_CLOCKWISE,
    )),
    @resource_dependencies begin
      @read
      normal_map::Texture
      @write
      (color => (0.08, 0.05, 0.1, 1.0))::Color
    end
  )
end

@testset "Texture drawing" begin
  vdata = [
    TextureCoordinates(Vec2(-0.5, 0.5), Vec2(0.0, 0.0)),
    TextureCoordinates(Vec2(-0.5, -0.5), Vec2(0.0, 1.0)),
    TextureCoordinates(Vec2(0.5, 0.5), Vec2(1.0, 0.0)),
    TextureCoordinates(Vec2(0.5, -0.5), Vec2(1.0, 1.0)),
  ]
  invocation = texture_invocation(device, vdata, color)
  data = render_graphics(device, graphics_node(invocation))
  save_test_render("distorted_normal_map.png", data, 0x9eda4cb9b969b269)
end;
