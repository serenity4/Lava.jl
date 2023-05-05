struct TextData
  positions::DeviceAddress # Vector{Vec2}, by vertex index
  curves::DeviceAddress # Vector{Vec3}, by curve index (from glyph_ranges)
  glyph_ranges::DeviceAddress # Vector{UnitRange{UInt32}} # by quad index, i.e. (vertex index) ÷ 4
  camera::PinholeCamera
  color::Vec3
end

function glyph_quads(line::Line, segment::LineSegment, glyph_size, pen_position = (0, 0))
  positions = Vec2[]
  glyph_curves = Arr{3,Vec2}[]
  glyph_ranges = UnitRange{UInt32}[] # ranges are 0-based
  processed_glyphs = Dict{GlyphID,Int}() # to glyph data index
  n = length(segment.indices)
  glyph_data_indices = Vector{UInt32}(undef, n) # data indices are 0-based
  (; font, options) = segment
  scale = options.font_size / font.units_per_em
  for i in segment.indices
    position = line.positions[i]
    glyph_id = line.glyphs[i]
    glyph = @something(font[glyph_id], font[GlyphID(0)])
    (; header) = glyph
    min = Point(header.xmin, -header.ymin)
    max = Point(header.xmax, -header.ymax)
    set = PointSet(box(scale .* min .+ pen_position, scale .* max .+ pen_position), Point2)
    append!(positions, Vec2.(@view set.points[[1, 2, 3, 4]]))
    index = get!(processed_glyphs, glyph_id) do
      start = lastindex(glyph_curves)
      append!(glyph_curves, Arr{3,Vec2}.(broadcast.(Vec2, curves(glyph) .* scale)))
      stop = lastindex(glyph_curves) - 1
      push!(glyph_ranges, start:stop)
      length(processed_glyphs)
    end
    glyph_data_indices[i] = index
    pen_position = pen_position .+ position.advance .* scale
  end
  (; positions, glyph_curves, glyph_ranges, glyph_data_indices)
end

function text_invocation_data(prog, line::Line, segment::LineSegment, start::Vec2, camera::PinholeCamera)
  pen_position = start
  (; positions, glyph_curves, glyph_ranges, glyph_data_indices) = glyph_quads(line, segment, pen_position)
  (; r, g, b) = segment.style.color
  color = (r, g, b)
  data = @invocation_data prog begin
    b1 = @block positions
    b2 = @block glyph_curves
    b3 = @block glyph_ranges
    @block TextData(@address(b1), @address(b2), @address(b3), camera, color)
  end
end

function text_vert(position, frag_position, frag_quad_index, index, data_address)
  data = @load data_address::TextData
  xy = @load data.positions[index]::Vec2
  frag_position[] = xy
  frag_quad_index[0U] = index ÷ 4U
  position.xyz = project(Vec3(xy..., 1F), data.camera)
  position.w = 1F
end

function text_frag(out_color, position, quad_index, data_address)
  quad_index = quad_index[0U]
  (; glyph_ranges, curves, color) = @load data_address::TextData
  range = @load glyph_ranges[quad_index]::UnitRange{UInt32}
  out_color.rgb = color
  out_color.a = 1F # intensity(position, curves, range)
end

function text_program(device)
  vert = @vertex device text_vert(::Vec4::Output{Position}, ::Vec2::Output, ::Vec{2,UInt32}::Output, ::UInt32::Input{VertexIndex}, ::DeviceAddressBlock::PushConstant)
  frag = @fragment device text_frag(::Vec4::Output, ::Vec2::Input, ::Vec{2,UInt32}::Input{@Flat}, ::DeviceAddressBlock::PushConstant)

  Program(vert, frag)
end

function draw_text(device, line::Line, segment::LineSegment, start, camera::PinholeCamera; options = FontOptions(ShapingOptions(tag"latn", tag"fra "), 12), prog = text_program(device))
  data = text_invocation_data(prog, line, segment, start, camera)
  strip = TriangleList(TriangleStrip(1:4))
  indices = Int[]
  for i in eachindex(line.glyphs)
    for triangle in strip.indices
      append!(indices, triangle .+ 4(i - 1))
    end
  end
  graphics_command(
    DrawIndexed(indices),
    prog,
    data,
    RenderTargets(color),
    RenderState(),
    setproperties(
      ProgramInvocationState(),
      (;
        triangle_orientation = Vk.FRONT_FACE_COUNTER_CLOCKWISE,
      ),
    ),
    @resource_dependencies @write (color => (0.08, 0.05, 0.1, 1.0))::Color
  )
end

@testset "Text rendering" begin
  font = OpenTypeFont(font_file("juliamono-regular.ttf"));
  text = Text("The brown fox jumps over the lazy dog.", TextOptions())
  options = FontOptions(ShapingOptions(tag"latn", tag"fra "), 1/37)
  line = only(lines(text, [font => options]))
  segment = only(line.segments)
  start = Vec2(0.1, 0.1)
  quads = glyph_quads(line, segment, start)
  camera = PinholeCamera(focal_length = 0.35F, transform = Transform(translation = (0.3, -0.3, 0), scaling = (1, -1, 1)))
  any(<(0), project(Vec3(start..., 1), camera))
  draw = draw_text(device, line, segment, start, camera)
  data = render_graphics(device, draw)
  save_test_render("text.png", data)
end
