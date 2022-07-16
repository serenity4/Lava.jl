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

function texture_program(device)
  vert = @vertex device.spirv_features texture_vert(::Output::Vec{2,Float32}, ::Output{Position}::Vec{4,Float32}, ::Input{VertexIndex}::UInt32, ::PushConstant::DrawData)
  frag = @fragment device.spirv_features texture_frag(
    ::Output::Vec{4,Float32},
    ::Input::Vec{2,Float32},
    ::PushConstant::DrawData,
    ::UniformConstant{DescriptorSet = 0, Binding = 3}::Arr{2048,SPIRV.SampledImage{SPIRV.Image{Float32,SPIRV.Dim2D,0,false,false,1,SPIRV.ImageFormatRgba16f}}})
  Program(device, vert, frag)
end

function program_2(device, vdata, color, uv::Vec{2,Float32} = Vec2(0.1, 1.0))
  rg = RenderGraph(device)

  normal = load(texture_file("normal.png"))
  normal = convert(Matrix{RGBA{Float16}}, normal)
  normal_map = image(device, normal; usage = Vk.IMAGE_USAGE_SAMPLED_BIT)
  normal_map = PhysicalImage(normal_map)

  graphics = RenderNode(render_area = RenderArea(Lava.dims(color)...), stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT | Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)

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
  set_material(rec, rg, MaterialDataTexture(
    uv, # uv scaling coefficients
    request_descriptor_index(rg, graphics, tex),
  ))
  draw(graphics, rec, rg, vdata, collect(1:4), color)

  rg
end

@testset "Texture drawing" begin
  vdata = [
    VertexDataTexture(Vec2(-0.5, 0.5), Vec2(0.0, 0.0)),
    VertexDataTexture(Vec2(-0.5, -0.5), Vec2(0.0, 1.0)),
    VertexDataTexture(Vec2(0.5, 0.5), Vec2(1.0, 0.0)),
    VertexDataTexture(Vec2(0.5, -0.5), Vec2(1.0, 1.0)),
  ]
  rg = program_2(device, vdata, pcolor)

  render(rg)
  data = collect(RGBA{Float16}, color.view.image, device)
  save_test_render("distorted_normal_map.png", data, 0x9eda4cb9b969b269)
end;
