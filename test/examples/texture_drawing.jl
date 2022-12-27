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

function program_2(device, vdata, color, uv::Vec{2,Float32} = Vec2(0.1, 1.0); prog = texture_program(device), normal_map = nothing)
  rg = RenderGraph(device)
  normal_map = @something(normal_map, read_normal_map(device))
  graphics = RenderNode(render_area = RenderArea(color.data.view.image.dims...), stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT | Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)

  @add_resource_dependencies rg begin
    (color => (0.08, 0.05, 0.1, 1.0))::Color = graphics(normal_map::Texture)
  end

  rec = StatefulRecording()
  set_program(rec, texture_program(device))
  set_invocation_state(rec, setproperties(invocation_state(rec), (;
    primitive_topology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
    triangle_orientation = Vk.FRONT_FACE_COUNTER_CLOCKWISE,
  )))
  tex = Texture(normal_map, setproperties(DEFAULT_SAMPLING, (magnification = Vk.FILTER_LINEAR, minification = Vk.FILTER_LINEAR)))
  coords_ptr = allocate_data(rec, rg, vdata)
  drawing_ptr = allocate_data(rec, rg, TextureDrawing(
    uv, # uv scaling coefficients
    request_index!(device, texture_descriptor(tex, graphics)),
  ))
  set_data(rec, rg, TextureData(coords_ptr, drawing_ptr))
  draw(graphics, rec, collect(1:4), color)

  rg
end

@testset "Texture drawing" begin
  vdata = [
    TextureCoordinates(Vec2(-0.5, 0.5), Vec2(0.0, 0.0)),
    TextureCoordinates(Vec2(-0.5, -0.5), Vec2(0.0, 1.0)),
    TextureCoordinates(Vec2(0.5, 0.5), Vec2(1.0, 0.0)),
    TextureCoordinates(Vec2(0.5, -0.5), Vec2(1.0, 1.0)),
  ]
  rg = program_2(device, vdata, color)

  render!(rg)
  data = collect(RGBA{Float16}, color.data.view.image, device)
  save_test_render("distorted_normal_map.png", data, 0x9eda4cb9b969b269)
end;
