function xcb_surface(instance, win::XCBWindow)
  handle = unwrap(Vk.create_xcb_surface_khr(instance, Vk.XcbSurfaceCreateInfoKHR(win.conn.h, win.id)))
  Surface(handle, win)
end

function render_rectangle(device, image, uv)
  vdata = [
    VertexDataTexture(Vec2(-0.5, 0.5), Vec2(0.0, 0.0)),
    VertexDataTexture(Vec2(-0.5, -0.5), Vec2(0.0, 1.0)),
    VertexDataTexture(Vec2(0.5, 0.5), Vec2(1.0, 0.0)),
    VertexDataTexture(Vec2(0.5, -0.5), Vec2(1.0, 1.0)),
  ]
  rg = program_2(device, vdata, PhysicalAttachment(Attachment(View(image), WRITE)), uv)
  command_buffer = Lava.request_command_buffer(device)
  baked = render(command_buffer, rg)
  Lava.transition_layout(command_buffer, image, Vk.IMAGE_LAYOUT_PRESENT_SRC_KHR)
  SubmissionInfo(command_buffers = [Vk.CommandBufferSubmitInfo(command_buffer)], release_after_completion = [baked])
end

@testset "Presenting to XCB" begin
  conn = Connection()
  screen = current_screen(conn)
  win = XCBWindow(conn, screen; x=0, y=1000, border_width=50, window_title="Test window", icon_title="Test", attributes=[XCB.XCB_CW_BACK_PIXEL], values=[screen.black_pixel])
  swapchain = test_validation_msg(x -> @test isempty(x)) do
    Swapchain(device, xcb_surface(instance, win), Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT | Vk.IMAGE_USAGE_TRANSFER_SRC_BIT)
  end
  cycle = FrameCycle(device, swapchain)
  set_presentation_queue(device, [swapchain.surface])
  cycle_f(x, t) = render_rectangle(device, x, Vec2(0.1 + t * 0.9, 0.5cos(t)))
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
  XCB.set_extent(win, XCB.extent(win) .+ 50)
  cycle_render(cycle, t)
  # `ERROR_OUT_OF_DATE_KHR` is triggered on the second call.
  cycle_render(cycle, t)
  @test any(!iszero, collect(BGRA{N0f8}, cycle))

  finalize(win)
end;
