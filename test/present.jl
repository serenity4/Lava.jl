function xcb_surface(instance, win::XCBWindow)
  handle = unwrap(Vk.create_xcb_surface_khr(instance, Vk.XcbSurfaceCreateInfoKHR(win.conn.h, win.id)))
  Surface(handle, win)
end

function render_rectangle(device, color, uv, image)
  rg = program_2(device, vdata, attachment_resource(ImageView(color), WRITE), uv; image)
  command_buffer = Lava.request_command_buffer(device)
  baked = render!(rg, command_buffer)
  Lava.transition_layout(command_buffer, color, Vk.IMAGE_LAYOUT_PRESENT_SRC_KHR)
  SubmissionInfo(command_buffers = [Vk.CommandBufferSubmitInfo(command_buffer)], release_after_completion = [baked])
end

@testset "Presenting to XCB" begin
  wm = XWindowManager()
  screen = current_screen(wm)
  win = XCBWindow(wm; screen, x=0, y=1000, border_width=50, window_title="Test window", icon_title="Test", attributes=[XCB.XCB_CW_BACK_PIXEL], values=[screen.black_pixel])
  swapchain = test_validation_msg(x -> @test isempty(x)) do
    Swapchain(device, xcb_surface(instance, win), Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT | Vk.IMAGE_USAGE_TRANSFER_SRC_BIT)
  end
  cycle = FrameCycle(device, swapchain)
  image = read_normal_map(device)
  set_presentation_queue(device, [swapchain.surface])

  vdata = [
    TextureCoordinates(Vec2(-0.5, 0.5), Vec2(0.0, 0.0)),
    TextureCoordinates(Vec2(-0.5, -0.5), Vec2(0.0, 1.0)),
    TextureCoordinates(Vec2(0.5, 0.5), Vec2(1.0, 0.0)),
    TextureCoordinates(Vec2(0.5, -0.5), Vec2(1.0, 1.0)),
  ]
  draw = draw_rectangle(device, vdata, color)

  cycle_f(color, t) = render_rectangle(device, color, Vec2(0.1 + t * 0.9, 0.5cos(t)), image)
  cycle_render(cycle, t) = wait(cycle!(Base.Fix2(cycle_f, t), cycle))

  test_validation_msg(x -> @test isempty(x)) do
    Δt = 0.1
    cycle_render(cycle, Δt)
    t0 = time()
    t = time()
    Δt = t - t0
    while Δt < 1
      cycle_render(cycle, Δt)
      Δt = time() - t
    end
  end

  t = 0.5
  Lava.recreate!(cycle)
  @test all(iszero, collect(BGRA{N0f8}, cycle))
  cycle_render(cycle, t)
  @test any(!iszero, collect(BGRA{N0f8}, cycle))
  resize(win, extent(win) .+ 50)
  cycle_render(cycle, t)
  # `ERROR_OUT_OF_DATE_KHR` is triggered on the second call.
  cycle_render(cycle, t)
  @test any(!iszero, collect(BGRA{N0f8}, cycle))

  finalize(win)
end;
