function xcb_surface(instance, win::XCBWindow)
  handle = unwrap(Vk.create_xcb_surface_khr(instance, Vk.XcbSurfaceCreateInfoKHR(win.conn.h, win.id)))
  Surface(handle, win)
end

function render_rectangle(device, image, uv)
  vdata = [
    (-0.5f0, 0.5f0, 0.0f0, 0.0f0),
    (-0.5f0, -0.5f0, 0.0f0, 1.0f0),
    (0.5f0, 0.5f0, 1.0f0, 0.0f0),
    (0.5f0, -0.5f0, 1.0f0, 1.0f0),
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
    Swapchain(device, xcb_surface(instance, win), Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
  end
  cycle = FrameCycle(device, swapchain)
  set_presentation_queue(device, [swapchain.surface])

  # Preserve render graph for a full cycle.
  preserve = Ref{ExecutionState}()
  test_validation_msg(x -> @test isempty(x)) do
    Δt = 0.1
    preserve[] = cycle!(x -> render_rectangle(device, x, Float32.((0.1 + Δt * 0.9, 0.5cos(Δt)))), cycle)
    t0 = time()
    t = time()
    Δt = t - t0
    while Δt < 1
      wait(preserve[])
      preserve[] = cycle!(x -> render_rectangle(device, x, Float32.((0.1 + Δt * 0.9, 0.5cos(Δt)))), cycle)
      Δt = time() - t
    end
    wait(preserve[])
  end
  finalize(win)
end
