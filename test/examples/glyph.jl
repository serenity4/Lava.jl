# Uses the technique from GPU-Centered Font Rendering Directly from Glyph Outlines, E. Lengyel, 2017.
# Note: this technique is patented in the US until 2038: https://patents.google.com/patent/US10373352B1/en.

function intensity(curve_points, pixel_per_em)
  ((x₁, y₁), (x₂, y₂), (x₃, y₃)) = curve_points
  T = typeof(x₁)
  res = zero(T)

  # Cast a ray in the X direction.
  code = classify_bezier_curve((y₁, y₂, y₃))
  bezier = BezierCurve(3)
  if !iszero(code)
    (t₁, t₂) = compute_roots(y₁ - 2y₂ + y₃, y₁ - y₂, y₁)
    if !isnan(t₁)
      val = zero(res)
      code & 0x0001 == 0x0001 && (val += winding_contribution(pixel_per_em, first(bezier(t₁, curve_points))))
      code > 0x0001 && (val -= winding_contribution(pixel_per_em, first(bezier(t₂, curve_points))))
      res += val
    end
  end

  # Cast a ray in the Y direction.
  code = classify_bezier_curve((x₁, x₂, x₃))
  if !iszero(code)
    (t₁, t₂) = compute_roots(x₁ - 2x₂ + x₃, x₁ - x₂, x₁)
    if !isnan(t₁)
      val = zero(res)
      code & 0x0001 == 0x0001 && (val += winding_contribution(pixel_per_em, last(bezier(t₁, curve_points))))
      code > 0x0001 && (val -= winding_contribution(pixel_per_em, last(bezier(t₂, curve_points))))
      res -= val
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

function intensity(position, curves::DeviceAddress, range)
  res = 0F
  for i in range
    curve_points = @load curves[i]::Arr{3,Vec2}
    res += intensity(curve_points .- Ref(position), 36F)
  end
  sqrt(abs(res))
end

struct GlyphData
  positions::DeviceAddress # Vector{Vec2}
  curves::DeviceAddress # Vector{Vec3}
  range::UnitRange{UInt32}
  color::Vec3
end

function glyph_vert(position, frag_position, index, data_address)
  data = @load data_address::GlyphData
  position.xy = @load data.positions[index]::Vec2
  frag_position[] = position.xy
end

function glyph_frag(out_color, position, data_address)
  (; range, curves, color) = @load data_address::GlyphData
  out_color.rgb = color
  out_color.a = intensity(position, curves, range)
end

function glyph_program(device)
  vert = @vertex device glyph_vert(::Vec4::Output{Position}, ::Vec2::Output, ::UInt32::Input{VertexIndex}, ::DeviceAddressBlock::PushConstant)
  frag = @fragment device glyph_frag(::Vec4::Output, ::Vec2::Input, ::DeviceAddressBlock::PushConstant)

  Program(vert, frag)
end

"""
Remap a value from `(low1, high1)` to `(low2, high2)`.
"""
function remap(value, low1, high1, low2, high2)
  low2 + (value - low1) * (high2 - low2) / (high1 - low1)
end

function draw_glyph(device, vdata, glyph, glyph_color, color; prog = glyph_program(device))
  curves = map(x -> Arr{3,Vec2}(Vec2.(broadcast.(remap, x, 0.0, 1.0, -0.9, 0.9))), curves_normalized(glyph))
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
  save_test_render("glyph.png", data)
end

nothing;

#= Prototyping area

function intensity(point, glyph, font_size)
    res = sum(curves_normalized(glyph)) do p
        intensity(p .- Ref(point), float(font_size))
    end
    sqrt(abs(res))
end

function intensity2(curve_points, pixel_per_em)
    VT = eltype(curve_points)
    FT = eltype(VT)
    res = zero(FT)
    for coord in (1, 2)
        (x̄₁, x̄₂, x̄₃) = getindex.(curve_points, 3 - coord)
        code = classify_bezier_curve((x̄₁, x̄₂, x̄₃))
        rshift = ifelse(x̄₁ > 0, 1 << 1, 0) + ifelse(x̄₂ > 0, 1 << 2, 0) + ifelse(x̄₃ > 0, 1 << 3, 0)
        code = (0x2e74 >> rshift) & 0x0003
        # Terminate early if we know there are no roots.
        iszero(code) && continue
        (t₁, t₂) = compute_roots(x̄₁ - 2x̄₂ + x̄₃, x̄₁ - x̄₂, x̄₁)
        # In classes C and F, there may be no real roots.
        isnan(t₁) && continue
        bezier = BezierCurve(3)
        val = zero(FT)
        if code & 0x0001 == 0x0001 # `code` is 0x0001 or 0x0003
            val += saturated_softmax(pixel_per_em, bezier(t₁, curve_points)[coord])
            # val += smoothstep(-one(FT)/pixel_per_em, one(FT)/pixel_per_em, bezier(t₁, curve_points)[coord])
        end
        if code > 0x0001 # `code` is 0x0002 or 0x0003
            val -= saturated_softmax(pixel_per_em, bezier(t₂, curve_points)[coord])
            # val -= smoothstep(-one(FT)/pixel_per_em, one(FT)/pixel_per_em, bezier(t₂, curve_points)[coord])
        end
        res += val * (coord == 1 ? 1 : -1)
    end
    res
end

function plot_outline(glyph)
    cs = curves(glyph)
    p = plot()
    for (i, curve) in enumerate(cs)
        for (i, point) in enumerate(curve)
            color = i == 1 ? :blue : i == 2 ? :cyan : :green
            scatter!(p, [point[1]], [point[2]], legend=false, color=color)
        end
        points = BezierCurve().(0:0.1:1, Ref(curve))
        curve_color = UInt8[255 - Int(floor(i / length(cs) * 255)), 40, 40]
        plot!(p, first.(points), last.(points), color=string('#', bytes2hex(curve_color)))
    end
    p
end

function render_glyph(font, glyph, font_size)
    step = 0.01
    n = Int(inv(step))
    xs = 0:step:1
    ys = 0:step:1

    grid = map(xs) do x
        map(ys) do y
            Point(x, y)
        end
    end

    grid = hcat(grid...)

    is = map(grid) do p
        try
            intensity(p, glyph, font_size)
        catch e
            if e isa DomainError
                NaN
            else
                rethrow(e)
            end
        end
    end
    @assert !all(iszero, is)

    p = heatmap(is)
    xticks!(p, 1:n ÷ 10:n, string.(xs[1:n ÷ 10:n]))
    yticks!(p, 1:n ÷ 10:n, string.(ys[1:n ÷ 10:n]))
end

render_glyph(font, char::Char, font_size) = render_glyph(font, font[char], font_size)

using Plots

font = OpenTypeFont(font_file("juliamono-regular.ttf"));

glyph = font.glyphs[563]

glyph = font.glyphs[64]
plot_outline(glyph)
render_glyph(font, glyph, 12)

glyph = font.glyphs[75]
plot_outline(glyph)
render_glyph(font, glyph, 12)

glyph = font.glyphs[13]
plot_outline(glyph)
render_glyph(font, glyph, 12)

render_glyph(font, '€', 12)

render_glyph(font, 'A', 12)

=#
