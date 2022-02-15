struct VertexDataTexture
  pos::Vec{2,Float32}
  uv::Vec{2,Float32}
end

function texture_vert(uv, position, index, dd)
  vd = Pointer{Vector{VertexDataTexture}}(dd.vertex_data)[index]
  (; pos) = vd
  position[] = Vec(pos.x, pos.y, 0F, 1F)
  uv[] = vd.uv
end

struct MaterialDataTexture
  uv_scaling::Vec{2,Float32}
  img_index::UInt32
end

function texture_frag(out_color, uv, dd, images)
  md = Pointer{MaterialDataTexture}(dd.material_data)[]
  (; uv_scaling, img_index) = md
  texcolor = images[img_index](uv * uv_scaling)
  out_color[] = Vec(texcolor.r, texcolor.g, texcolor.b, 1F)
end

function program_2(device, vdata)
  vert_interface = ShaderInterface(
    storage_classes = [SPIRV.StorageClassOutput, SPIRV.StorageClassOutput, SPIRV.StorageClassInput, SPIRV.StorageClassPushConstant],
    variable_decorations = dictionary([
      1 => dictionary([SPIRV.DecorationLocation => [0U]]),
      2 => dictionary([SPIRV.DecorationBuiltIn => [SPIRV.BuiltInPosition]]),
      3 => dictionary([SPIRV.DecorationBuiltIn => [SPIRV.BuiltInVertexIndex]]),
    ]),
    features = device.spirv_features,
  )

  frag_interface = ShaderInterface(
    execution_model = SPIRV.ExecutionModelFragment,
    storage_classes = [SPIRV.StorageClassOutput, SPIRV.StorageClassInput, SPIRV.StorageClassPushConstant, SPIRV.StorageClassUniformConstant],
    variable_decorations = dictionary([
      1 => dictionary([SPIRV.DecorationLocation => [0U]]),
      2 => dictionary([SPIRV.DecorationLocation => [0U]]),
      4 => dictionary([SPIRV.DecorationDescriptorSet => [0U], SPIRV.DecorationBinding => [3U]]),
    ]),
    features = device.spirv_features,
  )

  vert_shader = @shader vert_interface texture_vert(::Vec{2,Float32}, ::Vec{4,Float32}, ::UInt32, ::DrawData)
  frag_shader = @shader frag_interface texture_frag(
    ::Vec{4,Float32},
    ::Vec{2,Float32},
    ::DrawData,
    ::Arr{2048,SampledImage{SPIRV.Image{Float32,SPIRV.Dim2D,0,false,false,1,SPIRV.ImageFormatRgba16f}}},
  )
  prog = Program(device, vert_shader, frag_shader)

  fg = FrameGraph(device)
  add_color_attachment(fg)

  normal = load(texture_file("normal.png"))
  normal = convert(Matrix{RGBA{Float16}}, normal)
  normal_map = image(device, normal, Vk.FORMAT_R16G16B16A16_SFLOAT; usage = Vk.IMAGE_USAGE_SAMPLED_BIT)
  register(fg.frame, :normal_map, normal_map)
  add_resource!(fg, :normal_map, ImageResourceInfo(Lava.format(normal_map)))

  add_pass!(fg, :main, RenderPass((0, 0, 1920, 1080))) do rec
    set_program(rec, prog)
    ds = draw_state(rec)
    set_draw_state(rec, @set ds.program_state.primitive_topology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP)
    set_material(rec,
      (0.1f0, 1.0f0), # uv scaling coefficients
      Texture(:normal_map, setproperties(DEFAULT_SAMPLING, (magnification = Vk.FILTER_LINEAR, minification = Vk.FILTER_LINEAR))),
    )
    draw(rec, TargetAttachments([:color]), vdata, collect(1:4))
  end

  usage = @resource_usages begin
    color::Color = main(normal_map::Texture)
  end
  add_resource_usage!(fg, usage)
  clear_attachments(fg, :main, [:color => (0.08, 0.05, 0.1, 1.0)])
  fg
end

@testset "Texture drawing" begin
  vdata = [
    (-0.5f0, -0.5f0, 0.0f0, 0.0f0),
    (0.5f0, -0.5f0, 1.0f0, 0.0f0),
    (-0.5f0, 0.5f0, 0.0f0, 1.0f0),
    (0.5f0, 0.5f0, 1.0f0, 1.0f0),
  ]
  fg = program_2(device, vdata)
  @test wait(render(fg))
  data = collect(RGBA{Float16}, image(fg.frame.resources[:color].data), device)
  save_test_render("distorted_normal_map.png", data, 0x9eda4cb9b969b269)
end
