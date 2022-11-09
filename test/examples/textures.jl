struct TextureCoordinates
  pos::Vec{2,Float32}
  uv::Vec{2,Float32}
end

GeometryExperiments.location(coords::TextureCoordinates) = coords.pos
GeometryExperiments.vertex_attribute(coords::TextureCoordinates) = coords
Base.:(+)(x::TextureCoordinates, y::TextureCoordinates) = TextureCoordinates(x.pos + y.pos, x.uv + y.uv)
Base.:(*)(coords::TextureCoordinates, w::Real) = TextureCoordinates(coords.pos .* w, coords.uv .* w)
