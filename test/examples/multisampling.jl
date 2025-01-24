function draw_triangle_multisampled(device, vdata, color; prog = rectangle_program(device), samples)
  dependencies = if isnothing(samples)
    @resource_dependencies @write (color => (0.08, 0.05, 0.1, 1.0))::Color
  elseif samples == :four
    @resource_dependencies @write (color * 4 => (0.08, 0.05, 0.1, 1.0))::Color
  else
    @resource_dependencies @write (color * samples => (0.08, 0.05, 0.1, 1.0))::Color
  end

  invocation_data = @invocation_data prog @block vdata
  graphics_command(
    DrawIndexed(1:3),
    prog,
    invocation_data,
    RenderTargets(color),
    RenderState(),
    ProgramInvocationState(),
    dependencies,
  )
end

@testset "Multisampled triangle" begin
  vdata = [
    PosColor(Vec2(0.0, -0.8), Vec3(1.0, 0.0, 0.0)),
    PosColor(Vec2(0.5, 0.8), Vec3(0.0, 0.0, 1.0)),
    PosColor(Vec2(-0.5, 0.8), Vec3(0.0, 1.0, 0.0)),
  ]

  for samples in (4, :four, nothing)
    draw = draw_triangle_multisampled(device, vdata, color_ms; samples)
    data = render_graphics(device, draw)
    save_test_render("triangle_multisampled.png", data, 0x48ec4de6d479aae0)
  end
end;
