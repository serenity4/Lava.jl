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
    ::UniformConstant{DescriptorSet = 0U, Binding = 3U}::Arr{2048,SPIRV.SampledImage{SPIRV.Image{Float32,SPIRV.Dim2D,0,false,false,1,SPIRV.ImageFormatRgba16f}}})
  Program(device, vert, frag)
end

function program_2(device, vdata, color, uv::NTuple{2,Float32} = (0.1f0, 1.0f0))
  rg = RenderGraph(device)

  normal = load(texture_file("normal.png"))
  normal = convert(Matrix{RGBA{Float16}}, normal)
  normal_map = wait(image(device, Vk.FORMAT_R16G16B16A16_SFLOAT, normal; usage = Vk.IMAGE_USAGE_SAMPLED_BIT))
  normal_map = PhysicalImage(normal_map)

  graphics = RenderNode(render_area = RenderArea(Lava.dims(color)...), stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT | Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT) do rec
    set_program(rec, texture_program(device))
    ds = draw_state(rec)
    @reset ds.program_state.primitive_topology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP
    @reset ds.program_state.triangle_orientation = Vk.FRONT_FACE_COUNTER_CLOCKWISE
    set_draw_state(rec, ds)
    set_material(rec,
      uv, # uv scaling coefficients
      Texture(rec, normal_map, setproperties(DEFAULT_SAMPLING, (magnification = Vk.FILTER_LINEAR, minification = Vk.FILTER_LINEAR))),
    )
    draw(rec, vdata, collect(1:4), color; alignment = 4)
  end

  @add_resource_dependencies rg begin
    (color => (0.08, 0.05, 0.1, 1.0))::Color = graphics(normal_map::Texture)
  end
end

@testset "Texture drawing" begin
  vdata = [
    (-0.5f0, 0.5f0, 0.0f0, 0.0f0),
    (-0.5f0, -0.5f0, 0.0f0, 1.0f0),
    (0.5f0, 0.5f0, 1.0f0, 0.0f0),
    (0.5f0, -0.5f0, 1.0f0, 1.0f0),
  ]
  rg = program_2(device, vdata, pcolor)

  @test wait(render(rg))
  data = collect(RGBA{Float16}, color.view.image, device)
  save_test_render("distorted_normal_map.png", data, 0x9eda4cb9b969b269)
end
