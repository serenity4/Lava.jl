struct TextureCoordinates
  pos::Vec{2,Float32}
  uv::Vec{2,Float32}
end

GeometryExperiments.location(coords::TextureCoordinates) = coords.pos
GeometryExperiments.vertex_attribute(coords::TextureCoordinates) = coords
Base.:(+)(x::TextureCoordinates, y::TextureCoordinates) = TextureCoordinates(x.pos + y.pos, x.uv + y.uv)
Base.:(*)(coords::TextureCoordinates, w::Real) = TextureCoordinates(coords.pos .* w, coords.uv .* w)

function read_normal_map(device)
  normal = convert(Matrix{RGBA{Float16}}, load(texture_file("normal.png")))
  normal_map = image_resource(device, normal; usage_flags = Vk.IMAGE_USAGE_SAMPLED_BIT)
end

function read_boid_image(device)
  boid = convert(Matrix{RGBA{Float16}}, load(texture_file("boid.png"))')
  image_resource(device, boid; usage_flags = Vk.IMAGE_USAGE_SAMPLED_BIT)
end
