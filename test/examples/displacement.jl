struct DisplacementData
  vertex_coordinates::DeviceAddress
  texture_index::DescriptorIndex
  camera::PinholeCamera
end

function displace_height_vert(position, index, textures, data_address)
  data = @load data_address::DisplacementData
  (; pos, uv) = @load data.vertex_coordinates[index + 1U]::TextureCoordinates
  height_map = textures[data.texture_index]
  height = height_map(uv, 0F).r
  global_position = Vec4(pos.x, pos.y, height, 1)
  position.xyz = project(global_position.xyz, data.camera)
  position.w = 1
end

displace_height_frag(out_color) = out_color[] = Vec4(0.2, 0.2, 0.2, 1.0)

function fresnel_shading(n₁, n₂, cosθ)
  R₀ = ((n₁ - n₂) / (n₁ + n₂))^2
  R₀ + (1 - R₀) * (1 - cosθ)^5
end

function scalar_displacement_program(device)
  vert = @vertex device displace_height_vert(
    ::Vec4::Output{Position},
    ::UInt32::Input{VertexIndex},
    ::Arr{2048,SPIRV.SampledImage{SPIRV.image_type(SPIRV.ImageFormatR16f,SPIRV.Dim2D,0,false,false,1)}}::UniformConstant{@DescriptorSet(0), @Binding(3)},
    ::DeviceAddressBlock::PushConstant,
  )
  frag = @fragment device displace_height_frag(::Vec4::Output)
  Program(vert, frag)
end

function draw_terrain(device, vmesh::VertexMesh, color::Resource, height_map::Resource, camera::PinholeCamera, prog = scalar_displacement_program(device))
  height_map_texture = texture_descriptor(Texture(height_map, setproperties(DEFAULT_SAMPLING, (magnification = Vk.FILTER_LINEAR, minification = Vk.FILTER_LINEAR))))
  invocation_data = @invocation_data prog begin
    @block DisplacementData(@address(@block vmesh.vertex_data), @descriptor(height_map_texture), camera)
  end
  graphics_command(
    DrawIndexed(foldl(vcat, vmesh.indices.indices); vertex_offset = 0),
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
      height_map::Texture
      @write
      (color => (0.08, 0.05, 0.1, 1.0))::Color
    end
  )
end

@testset "Scalar displacement" begin
  vdata = [
    TextureCoordinates(Vec2(-1.0, -1.0), Vec2(0.0, 0.0)),
    TextureCoordinates(Vec2(-1.0, 1.0), Vec2(0.0, 1.0)),
    TextureCoordinates(Vec2(1.0, 1.0), Vec2(1.0, 1.0)),
    TextureCoordinates(Vec2(1.0, -1.0), Vec2(1.0, 0.0)),
  ]
  mesh = Mesh{TextureCoordinates}(vdata, [(1, 2), (2, 3), (3, 4), (4, 1)], [[1, 4, 3, 2]])
  subdivide!(mesh, UniformSubdivision(2))
  triangulate!(mesh)
  @assert orientation(mesh) == FACE_ORIENTATION_COUNTERCLOCKWISE
  vmesh = VertexMesh(mesh)
  height_map = rand(MersenneTwister(0), Float32, 1920, 1080)
  height_map = image_resource(device, height_map; usage_flags = Vk.IMAGE_USAGE_SAMPLED_BIT, format = Vk.FORMAT_R32_SFLOAT)
  camera = PinholeCamera(focal_length = 2F)
  draw = draw_terrain(device, vmesh, color, height_map, camera)
  data = render_graphics(device, draw)
  h = save_test_render("displacement.png", data)
  @test isa(h, UInt64)
end;
