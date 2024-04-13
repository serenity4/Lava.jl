struct GaussianBlur
  σ::Float32
end

function gaussian_1d(t, σ)
  exp(-t^2 / 2σ^2) / sqrt(2 * πF * σ^2)
end
gaussian_2d((x, y), σ) = gaussian_1d(x, σ) * gaussian_1d(y, σ)

function compute_blur((; σ)::GaussianBlur, reference, uv)
  res = zero(Vec3)
  imsize = size(SPIRV.Image(reference), 0U)
  pixel_size = 1F ./ imsize # size of one pixel in UV coordinates.
  rx, ry = Int32.(min.(ceil.(3σ .* imsize), imsize))
  for i in -rx:rx
    for j in -ry:ry
      uv_offset = Vec2(i, j) .* pixel_size
      weight = gaussian_2d(uv_offset, σ) * 0.5(pixel_size[1]^2 + pixel_size[2]^2)
      sampled = reference(uv + uv_offset)
      color = sampled.rgb
      res .+= color * weight
    end
  end
  res
end

IT = image_type(SPIRV.ImageFormatRgba16f, SPIRV.Dim2D, 0, false, false, 1)

struct BlurData
  texture::TextureData
  blur::GaussianBlur
end

function blur_vert(uv, position, index, data_address::DeviceAddressBlock)
  data = @load data_address::BlurData
  coords = @load data.texture.coords[index + 1U]::TextureCoordinates
  (; pos) = coords
  position[] = Vec(pos.x, pos.y, 0F, 1F)
  uv[] = coords.uv
end

function blur_frag(out_color, uv, data_address, images)
  data = @load data_address::BlurData
  drawing = @load data.texture.drawing::TextureDrawing
  (; uv_scaling, img_index) = drawing
  reference = images[img_index]
  # color = reference(uv * uv_scaling)
  color = compute_blur(data.blur, reference, uv .* uv_scaling)
  out_color[] = Vec(color.r, color.g, color.b, 1F)
end

function blur_program(device)
  vert = @vertex device blur_vert(::Vec2::Output, ::Vec4::Output{Position}, ::UInt32::Input{VertexIndex}, ::DeviceAddressBlock::PushConstant)
  frag = @fragment device blur_frag(
    ::Vec4::Output,
    ::Vec2::Input,
    ::DeviceAddressBlock::PushConstant,
    ::Arr{2048,SPIRV.SampledImage{IT}}::UniformConstant{@DescriptorSet($GLOBAL_DESCRIPTOR_SET_INDEX), @Binding($BINDING_COMBINED_IMAGE_SAMPLER)})
  Program(vert, frag)
end

function blur_image(device, vdata, color, blur::GaussianBlur, uv_scale::Vec{2,Float32} = Vec2(0.1, 1.0); prog = blur_program(device), image = read_normal_map(device))
  image_texture = Texture(image, setproperties(DEFAULT_SAMPLING, (magnification = Vk.FILTER_LINEAR, minification = Vk.FILTER_LINEAR)))

  deps = @resource_dependencies begin
    @read
    image::Texture
    @write
    (color => (0.08, 0.05, 0.1, 1.0))::Color
  end

  invocation_data = @invocation_data prog begin
    tex1 = @block vdata
    tex2 = @block TextureDrawing(uv_scale, @descriptor(texture_descriptor(image_texture)))
    tex = TextureData(@address(tex1), @address(tex2))
    @block BlurData(tex, blur)
  end

  invocation_state = setproperties(ProgramInvocationState(), (;
    primitive_topology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
    triangle_orientation = Vk.FRONT_FACE_COUNTER_CLOCKWISE,
  ))

  graphics_command(
    DrawIndexed(1:4),
    prog,
    invocation_data,
    RenderTargets(color),
    RenderState(),
    invocation_state,
    deps
  )
end

@testset "Blur" begin
  vdata = [
    TextureCoordinates(Vec2(-0.5, -0.5), Vec2(0.0, 0.0)),
    TextureCoordinates(Vec2(-0.5, 0.5), Vec2(0.0, 1.0)),
    TextureCoordinates(Vec2(0.5, -0.5), Vec2(1.0, 0.0)),
    TextureCoordinates(Vec2(0.5, 0.5), Vec2(1.0, 1.0)),
  ]

  blur = GaussianBlur(0.01)
  reference = SPIRV.SampledImage(IT(zeros(32, 32)))
  @test compute_blur(blur, reference, zero(Vec2)) == zero(Vec3)
  uv_scale = Vec2(1.0, 1.0)
  command = blur_image(device, vdata, color, blur, uv_scale)
  data = render_graphics(device, command)
  save_test_render("blurred_normal_map.png", data, 0x5114a2d55a9aff00)
end;
