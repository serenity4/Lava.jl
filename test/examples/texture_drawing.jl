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
  texcolor = images[img_index](uv .* uv_scaling)
  out_color[] = Vec(texcolor.r, texcolor.g, texcolor.b, 1F)
end

function texture_program(device)
  vert = @vertex device texture_vert(::Vec2::Output, ::Vec4::Output{Position}, ::UInt32::Input{VertexIndex}, ::DeviceAddressBlock::PushConstant)
  frag = @fragment device texture_frag(
    ::Vec4::Output,
    ::Vec2::Input,
    ::DeviceAddressBlock::PushConstant,
    ::Arr{2048,SPIRV.SampledImage{SPIRV.image_type(SPIRV.ImageFormatRgba16f, SPIRV.Dim2D, 0, false, false, 1)}}::UniformConstant{DescriptorSet = 0, Binding = 3})
  Program(vert, frag)
end

function draw_texture(device, vdata, color; prog = texture_program(device), image = nothing, uv_scale = Vec2(0.1, 1.0))
  image = @something(image, read_normal_map(device))
  image_texture = texture_descriptor(Texture(image, setproperties(DEFAULT_SAMPLING, (magnification = Vk.FILTER_LINEAR, minification = Vk.FILTER_LINEAR))))
  invocation_data = @invocation_data prog begin
    b1 = @block vdata
    b2 = @block TextureDrawing(uv_scale, @descriptor image_texture)
    @block TextureData(@address(b1), @address(b2))
  end
  graphics_command(
    DrawIndexed(1:4),
    prog,
    invocation_data,
    RenderTargets(color),
    RenderState(),
    setproperties(ProgramInvocationState(), (;
      primitive_topology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
      triangle_orientation = Vk.FRONT_FACE_COUNTER_CLOCKWISE,
    )),
    @resource_dependencies begin
      @read
      image::Texture
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
  draw = draw_texture(device, vdata, color)
  data = render_graphics(device, draw)
  save_test_render("distorted_normal_map.png", data, 0x9eda4cb9b969b269)
end;
