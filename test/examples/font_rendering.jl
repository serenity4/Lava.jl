"""
Remap a value from `(low1, high1)` to `(low2, high2)`.
"""
function remap(value, low1, high1, low2, high2)
  low2 + (value - low1) * (high2 - low2) / (high1 - low1)
end

remap(value, from, to) = remap(value, from..., to...)
remap(values::AbstractArray, to; from = extrema(values)) = remap.(values, Ref(from), Ref(to))

remap(low1, high1, low2, high2) = x -> remap(x, low1, high1, low2, high2)

function program_3(device, positions, ppm)
  prog = Program(device, ShaderSource(shader_file("glyph.vert.spv")), ShaderSource(shader_file("glyph.frag.spv")))

  fg = FrameGraph(device)
  add_color_attachment(fg)

  font = OpenTypeFont(font_file("juliamono-regular.ttf"))
  curves = map(OpenType.curves(font['A'])) do curve
    Point{3,Point{2,Float32}}(map(curve) do point
      remapped = map(remap(0.0, 1.0, -0.9, 0.9), point)
      Point(remapped[1], -remapped[2])
    end)
  end
  curve_buffer = Buffer(device, curves)
  vdata = [(pos..., UInt32(0), UInt32(length(curves))) for pos in positions]
  register(fg.frame, :curve_buffer, curve_buffer)

  add_pass!(fg, :main, RenderPass((0, 0, 1920, 1080))) do rec
    set_program(rec, prog)
    ds = draw_state(rec)
    set_draw_state(rec, @set ds.program_invocation_state.primitive_topology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP)
    set_material(rec,
      RGBA{Float32}(1.0, 1.0, 0.0, 1.0), # text color
      device_address(curve_buffer),
      ppm, # pixel per em
      alignment = 8,
    )
    draw(rec, RenderTargets([:color]), vdata, collect(1:4); alignment = 8)
  end

  usage = @resource_dependencies begin
    color::Color = main()
  end
  add_resource_dependencies!(fg, usage)
  clear_attachments(fg, :main, [:color => (0.08, 0.05, 0.1, 1.0)])
  fg
end

@testset "Font rendering" begin
  positions = [
    (-1.0f0, -1.0f0),
    (1.0f0, -1.0f0),
    (-1.0f0, 1.0f0),
    (1.0f0, 1.0f0),
  ]
  fg = program_3(device, positions, 12.0f0)
  render!(fg)
  # FIXME: debug why font shader fails
  data = collect(image(fg.frame.resources[:color].data), device)
  save_test_render("glyph_A.png", data, 0x2a62b795abd45046)
end
