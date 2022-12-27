using Lava

struct GaussianBlur
  scale::Float32
  strength::Float32
end

function compute_blur(blur::GaussianBlur, reference, direction, uv)
  weights = Arr{Float32}(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216)
  tex_offset = 1F ./ size(SPIRV.Image(reference), 0U) .* blur.scale
  res = zero(Vec3)
  for i in eachindex(weights)
    vec = direction == 1U ? Vec2(tex_offset.x * i, 0F) : Vec2(0F, tex_offset.y * i)
    res .+= reference(uv .+ vec).rgb .* weights[i] .* blur.strength
    res .+= reference(uv .- vec).rgb .* weights[i] .* blur.strength
  end
  res
end

function compute_blur(blur::GaussianBlur, reference, uv)
  res = zero(Vec3)
  imsize = size(SPIRV.Image(reference), 0U)
  pixel_size = 1F ./ imsize # Size of one pixel in UV coordinates.
  radius = UInt32.(ceil.(imsize .* 0.5F .* blur.scale))
  rx, ry = radius
  for i in -rx:rx
    for j in -ry:ry
      uv_offset = Vec(i, j) .* pixel_size
      #TODO: Use blur strength parameter to control the falloff.
      weight = 0.25 .* rx .^ 2 .* ry .^ 2
      res .+= reference(uv .+ uv_offset).rgb .* weight
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
  coords = @load data.texture.coords[index]::TextureCoordinates
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
  color = compute_blur(data.blur, reference, uv * uv_scaling)
  out_color[] = Vec(color.r, color.g, color.b, 1F)
end

function blur_program(device)
  vert = @vertex device.spirv_features blur_vert(::Output::Vec2, ::Output{Position}::Vec4, ::Input{VertexIndex}::UInt32, ::PushConstant::DeviceAddressBlock)
  frag = @fragment device.spirv_features blur_frag(
    ::Output::Vec4,
    ::Input::Vec2,
    ::PushConstant::DeviceAddressBlock,
    ::UniformConstant{DescriptorSet = 0, Binding = 3}::Arr{2048,SPIRV.SampledImage{IT}})
  Program(device, vert, frag)
end

function blur_invocation(device, vdata, color, blur::GaussianBlur, uv_scale::Vec{2,Float32} = Vec2(0.1, 1.0); prog = blur_program(device), normal_map = read_normal_map(device))
  normal_map_texture = Texture(normal_map, setproperties(DEFAULT_SAMPLING, (magnification = Vk.FILTER_LINEAR, minification = Vk.FILTER_LINEAR)))

  deps = @resource_dependencies begin
    @read
    normal_map::Texture
    @write
    (color => (0.08, 0.05, 0.1, 1.0))::Color
  end

  invocation_data = @invocation_data begin
    tex1 = @block vdata
    tex2 = @block TextureDrawing(
      uv_scale, @descriptor(texture_descriptor(normal_map_texture))
    )
    tex = TextureData(@address(tex1), @address(tex2))
    @block BlurData(tex, blur)
  end

  invocation_state = setproperties(ProgramInvocationState(), (;
    primitive_topology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
    triangle_orientation = Vk.FRONT_FACE_COUNTER_CLOCKWISE,
  ))

  ProgramInvocation(
    prog,
    DrawIndexed(1:4),
    RenderTargets(color),
    invocation_data,
    RenderState(),
    invocation_state,
    deps
  )
end

@testset "Blur" begin
  vdata = [
    TextureCoordinates(Vec2(-0.5, 0.5), Vec2(0.0, 0.0)),
    TextureCoordinates(Vec2(-0.5, -0.5), Vec2(0.0, 1.0)),
    TextureCoordinates(Vec2(0.5, 0.5), Vec2(1.0, 0.0)),
    TextureCoordinates(Vec2(0.5, -0.5), Vec2(1.0, 1.0)),
  ]

  blur = GaussianBlur(0.1, 1.0)
  reference = SPIRV.SampledImage(IT(zeros(32, 32)))
  @test compute_blur(blur, reference, zero(Vec2)) == zero(Vec3)
  uv_scale = Vec2(1.0, 0.1)
  @test_skip begin
    invocation = blur_invocation(device, vdata, color, blur, uv_scale)
    graphics = RenderNode(render_area = RenderArea(1920, 1080), stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT | Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)
    push!(graphics.program_invocations, invocation)
    rg = RenderGraph(device)
    add_node!(rg, graphics)
    render!(rg)
    data = collect(RGBA{Float16}, color.data.view.image, device)
    @show save_test_render("blurred_normal_map.png", data)
  end
end;
