function xcb_surface(instance, win::XCBWindow)
  handle = unwrap(Vk.create_xcb_surface_khr(instance, Vk.XcbSurfaceCreateInfoKHR(win.conn.h, win.id)))
  Surface(handle, win)
end

function draw_on_screen(device, nodes, source, target)
  transfer = transfer_command(source, target)
  present = present_command(target)
  rg = RenderGraph(device, nodes)
  add_nodes!(rg, transfer, present)
  command_buffer = Lava.request_command_buffer(device)
  baked = render!(rg, command_buffer)
  SubmissionInfo(command_buffers = [Vk.CommandBufferSubmitInfo(command_buffer)], release_after_completion = [baked], queue_family = command_buffer.queue_family_index, signal_fence = Lava.fence(device))
end

@testset "Presenting to XCB" begin
  wm = XWindowManager()
  screen = current_screen(wm)
  win = XCBWindow(wm; screen, x=0, y=1000, border_width=50, window_title="Test window", icon_title="Test", attributes=[XCB.XCB_CW_BACK_PIXEL], values=[screen.black_pixel])
  swapchain = test_validation_msg(x -> @test isempty(x)) do
    Swapchain(device, xcb_surface(instance, win), Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_TRANSFER_DST_BIT)
  end
  cycle = FrameCycle(device, swapchain)
  color = attachment_resource(Vk.FORMAT_R16G16B16A16_SFLOAT, [1920, 1080])
  image = read_normal_map(device)
  prog = texture_program(device)
  set_presentation_queue(device, [swapchain.surface])
  vdata = [
    TextureCoordinates(Vec2(-0.5, 0.5), Vec2(0.0, 0.0)),
    TextureCoordinates(Vec2(-0.5, -0.5), Vec2(0.0, 1.0)),
    TextureCoordinates(Vec2(0.5, 0.5), Vec2(1.0, 0.0)),
    TextureCoordinates(Vec2(0.5, -0.5), Vec2(1.0, 1.0)),
  ]

  function _draw_on_screen(swapchain_image, t)
    uv_scale = Vec2(0.1 + t * 0.9, 0.5cos(t))
    nodes = [draw_texture(device, vdata, color; prog, uv_scale, image)]
    target = Resource(swapchain_image)
    draw_on_screen(device, nodes, color, target)
  end
  cycle_render(cycle, t) = wait(cycle!(Base.Fix2(_draw_on_screen, t), cycle))

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
  @test isa(collect(BGRA{N0f8}, cycle), Matrix{BGRA{N0f8}})
  cycle_render(cycle, t)
  @test any(!iszero, collect(BGRA{N0f8}, cycle))
  resize(win, extent(win) .+ 50)
  cycle_render(cycle, t)
  # `ERROR_OUT_OF_DATE_KHR` is triggered on the second call.
  cycle_render(cycle, t)
  @test any(!iszero, collect(BGRA{N0f8}, cycle))

  finalize(win)
end;
