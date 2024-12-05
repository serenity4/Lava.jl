# Uses the technique from GPU-Centered Font Rendering Directly from Glyph Outlines, E. Lengyel, 2017.
# Note: this technique is patented in the US until 2038: https://patents.google.com/patent/US10373352B1/en.

function intensity(bezier, pixel_per_em)
  ((x₁, y₁), (x₂, y₂), (x₃, y₃)) = bezier.points
  T = typeof(x₁)
  res = zero(T)

  # Cast a ray in the X direction.
  code = classify_bezier_curve((y₁, y₂, y₃))
  if !iszero(code)
    (t₁, t₂) = compute_roots(y₁ - 2y₂ + y₃, y₁ - y₂, y₁)
    if !isnan(t₁)
      code & 0x0001 == 0x0001 && (res += winding_contribution(pixel_per_em, first(bezier(t₁))))
      code > 0x0001 && (res -= winding_contribution(pixel_per_em, first(bezier(t₂))))
    end
  end

  # Cast a ray in the Y direction.
  code = classify_bezier_curve((x₁, x₂, x₃))
  if !iszero(code)
    (t₁, t₂) = compute_roots(x₁ - 2x₂ + x₃, x₁ - x₂, x₁)
    if !isnan(t₁)
      code & 0x0001 == 0x0001 && (res -= winding_contribution(pixel_per_em, last(bezier(t₁))))
      code > 0x0001 && (res += winding_contribution(pixel_per_em, last(bezier(t₂))))
    end
  end

  res
end

winding_contribution(pixel_per_em, value) = clamp(0.5F + pixel_per_em * value, 0F, 1F)

function classify_bezier_curve(points)
  (x₁, x₂, x₃) = points
  rshift = ifelse(x₁ > 0, 1 << 1, 0) + ifelse(x₂ > 0, 1 << 2, 0) + ifelse(x₃ > 0, 1 << 3, 0)
  (0x2e74 >> rshift) & 0x0003
end

function intensity(position, curves::DeviceAddress, range, pixel_per_em)
  res = 0F
  for i in range
    curve = BezierCurve((@load curves[i]::Arr{3,Vec2}) .- Ref(position))
    res += intensity(curve, pixel_per_em)
  end
  sqrt(abs(res))
end

struct GlyphData
  positions::DeviceAddress # Vector{Vec2}
  curves::DeviceAddress # Vector{Arr{3,Vec2}}
  range::UnitRange{UInt32}
  color::Vec3
end

function glyph_vert(position, frag_position, index, data_address)
  data = @load data_address::GlyphData
  position.xy = @load data.positions[index + 1U]::Vec2
  frag_position[] = position.xy
end

function glyph_frag(out_color, position, data_address)
  (; range, curves, color) = @load data_address::GlyphData
  @swizzle out_color.rgb = color
  @swizzle out_color.a = intensity(position, curves, range, 36F)
end

function glyph_program(device)
  vert = @vertex device glyph_vert(::Mutable{Vec4}::Output{Position}, ::Mutable{Vec2}::Output, ::UInt32::Input{VertexIndex}, ::DeviceAddressBlock::PushConstant)
  frag = @fragment device glyph_frag(::Mutable{Vec4}::Output, ::Vec2::Input, ::DeviceAddressBlock::PushConstant)

  Program(vert, frag)
end

function draw_glyph(device, vdata, glyph, glyph_color, color; prog = glyph_program(device))
  transform((x, y)) = Vec2(remap(x, glyph.header.xmin, glyph.header.xmax, 0, 1), remap(y, glyph.header.ymin, glyph.header.ymax, 0, 1))
  curves = map(ps -> Arr{3,Vec2}(transform.(ps)), OpenType.curves(glyph))
  data = @invocation_data prog begin
    b1 = @block vdata
    b2 = @block curves
    @block GlyphData(@address(b1), @address(b2), eachindex(curves), Vec3(glyph_color.r, glyph_color.g, glyph_color.b))
  end
  graphics_command(
    DrawIndexed(eachindex(vdata)),
    prog,
    data,
    RenderTargets(color),
    RenderState(),
    setproperties(
      ProgramInvocationState(),
      (;
        primitive_topology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
        triangle_orientation = Vk.FRONT_FACE_COUNTER_CLOCKWISE,
      ),
    ),
    @resource_dependencies @write (color => (0.08, 0.05, 0.1, 1.0))::Color
  )
end

@testset "Glyph rendering" begin
  font = OpenTypeFont(font_file("juliamono-regular.ttf"));
  vdata = [
    Vec2(-1, 1),
    Vec2(-1, -1),
    Vec2(1, 1),
    Vec2(1, -1),
  ]
  draw = draw_glyph(device, vdata, font['A'], RGB(0.6, 0.4, 1.0), color)
  data = render_graphics(device, draw)
  save_test_render("glyph.png", data, 0x29cebbf69bf12f45)
end
