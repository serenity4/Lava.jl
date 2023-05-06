struct TextData
  positions::DeviceAddress # Vector{Vec2}, by vertex index
  glyph_curves::DeviceAddress # Vector{Vec3}, by curve index (from glyph_ranges)
  glyph_indices::DeviceAddress # Vector{UInt32}, by quad index, i.e. (vertex index) ÷ 4
  glyph_ranges::DeviceAddress # Vector{UnitRange{UInt32}}, by glyph index
  glyph_origins::DeviceAddress # Vector{Vec2}, by quad index
  font_size::Float32
  camera::PinholeCamera
  color::Vec3
end

function glyph_quads(line::Line, segment::LineSegment, glyph_size, pen_position = (0, 0))
  positions = Vec2[]
  glyph_curves = Arr{3,Vec2}[]
  glyph_ranges = UnitRange{UInt32}[] # ranges are 0-based
  processed_glyphs = Dict{GlyphID,Int}() # to glyph index
  n = length(segment.indices)
  glyph_indices = UInt32[] # 0-based
  glyph_origins = Vec2[]
  (; font, options) = segment
  scale = options.font_size / font.units_per_em
  for i in segment.indices
    position = line.positions[i]
    origin = pen_position .+ position.origin .* scale
    pen_position = pen_position .+ position.advance .* scale
    glyph_id = line.glyphs[i]
    glyph = font[glyph_id]
    # Assume that the absence of a glyph means there is no glyph to draw.
    isnothing(glyph) && continue
    (; header) = glyph
    min = scale .* Point(header.xmin, header.ymin)
    max = scale .* Point(header.xmax, header.ymax)
    set = PointSet(box(min .+ origin, max .+ origin), Point2)
    append!(positions, Vec2.(set.points))
    index = get!(processed_glyphs, glyph_id) do
      start = lastindex(glyph_curves)
      append!(glyph_curves, Arr{3,Vec2}.(broadcast.(Vec2, curves(glyph) ./ font.units_per_em)))
      stop = lastindex(glyph_curves) - 1
      push!(glyph_ranges, start:stop)
      length(processed_glyphs) + 1
    end
    push!(glyph_indices, index - 1)
    push!(glyph_origins, Vec2(origin))
  end
  (; positions, glyph_curves, glyph_indices, glyph_ranges, glyph_origins)
end

function text_invocation_data(prog, line::Line, segment::LineSegment, start::Vec2, camera::PinholeCamera)
  pen_position = start
  (; positions, glyph_curves, glyph_indices, glyph_ranges, glyph_origins) = glyph_quads(line, segment, pen_position)
  (; r, g, b) = segment.style.color
  color = (r, g, b)
  data = @invocation_data prog begin
    b1 = @block positions
    b2 = @block glyph_curves
    b3 = @block glyph_indices
    b4 = @block glyph_ranges
    b5 = @block glyph_origins
    @block TextData(@address(b1), @address(b2), @address(b3), @address(b4), @address(b5), segment.options.font_size, camera, color)
  end
end

function text_vert(position, glyph_coordinates, frag_quad_index, index, data_address)
  data = @load data_address::TextData
  xy = @load data.positions[index]::Vec2
  quad_index = index ÷ 4U
  frag_quad_index.x = quad_index
  origin = @load data.glyph_origins[quad_index]::Vec2
  glyph_coordinates[] = xy - origin
  position.xyz = project(Vec3(xy..., 1F), data.camera)
  position.w = 1F
end

function text_frag(out_color, glyph_coordinates, quad_index, data_address)
  quad_index = quad_index.x
  (; glyph_curves, glyph_indices, glyph_ranges, glyph_origins, font_size, color) = @load data_address::TextData
  glyph_index = @load glyph_indices[quad_index]::UInt32
  range = @load glyph_ranges[glyph_index]::UnitRange{UInt32}
  out_color.rgb = color
  out_color.a = intensity(glyph_coordinates / font_size, glyph_curves, range, 60F)
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
    RenderTargets(color_ms),
    RenderState(),
    setproperties(
      ProgramInvocationState(),
      (;
        triangle_orientation = Vk.FRONT_FACE_COUNTER_CLOCKWISE,
      ),
    ),
    @resource_dependencies @write (color_ms => (0.08, 0.05, 0.1, 1.0))::Color
  )
end

@testset "Text rendering" begin
  font = OpenTypeFont(font_file("juliamono-regular.ttf"));
  options = FontOptions(ShapingOptions(tag"latn", tag"fra "), 1/37)
  camera = PinholeCamera(focal_length = 0.35F, transform = Transform(translation = (0.3, 0, 0), scaling = (1, 1, 1)))
  start = Vec2(0.1, 0.1)

  text = Text("The brown fox jumps over the lazy dog.", TextOptions())
  line = only(lines(text, [font => options]))
  segment = only(line.segments)
  quads = glyph_quads(line, segment, start)
  @test length(quads.glyph_indices) == count(!isspace, text.chars)
  @test length(quads.glyph_ranges) == count(x -> !isnothing(font[x]), unique(line.glyphs))
  any(<(0), project(Vec3(start..., 1), camera))
  draw = draw_text(device, line, segment, start, camera)
  data = render_graphics(device, draw)
  save_test_render("text.png", data, 0xe158233b14b89ab6)

  font = OpenTypeFont(font_file("NotoSerifLao.ttf"));
  options = FontOptions(ShapingOptions(tag"lao ", tag"dflt"; enabled_features = Set([tag"aalt"])), 1/10)
  text = Text("ກີບ ສົ \ue99\ueb5\uec9", TextOptions())
  line = only(lines(text, [font => options]))
  segment = only(line.segments)
  draw = draw_text(device, line, segment, start, camera)
  data = render_graphics(device, draw)
  save_test_render("text_2.png", data, 0xf7f6e947afd362b3)
end
