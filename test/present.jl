@testset "Presenting to XCB" begin
  wm = XWindowManager()
  win = XCBWindow(wm; x=0, y=1000, border_width=50, window_title="Test window", icon_title="Test", attributes=[XCB.XCB_CW_BACK_PIXEL], values=[zero(UInt32)])
  cycle = test_validation_msg(() -> FrameCycle(device, Surface(instance, win)), x -> @test isempty(x))
  color = attachment_resource(RGBA{Float16}, [1920, 1080])
  image = read_normal_map(device)
  prog = texture_program(device)
  vdata = [
    TextureCoordinates(Vec2(-0.5, 0.5), Vec2(0.0, 0.0)),
    TextureCoordinates(Vec2(-0.5, -0.5), Vec2(0.0, 1.0)),
    TextureCoordinates(Vec2(0.5, 0.5), Vec2(1.0, 0.0)),
    TextureCoordinates(Vec2(0.5, -0.5), Vec2(1.0, 1.0)),
  ]

  function draw(swapchain_image, t)
    uv_scale = Vec2(0.1 + t * 0.9, 0.5cos(t))
    nodes = [draw_texture(device, vdata, color; prog, uv_scale, image)]
    draw_and_prepare_for_presentation(device, nodes, color, swapchain_image)
  end
  draw_and_present(cycle, t) = wait(cycle!(image -> draw(image, t), cycle))

  test_validation_msg(x -> @test isempty(x)) do
    Δt = 0.1
    draw_and_present(cycle, Δt)
    t0 = time()
    t = time()
    Δt = t - t0
    while Δt < 1
      draw_and_present(cycle, Δt)
      Δt = time() - t
    end
  end

  Lava.recreate!(cycle)
  @test isa(collect(BGRA{N0f8}, cycle), Matrix{BGRA{N0f8}})
  draw_and_present(cycle, 0.5)
  @test any(!iszero, collect(BGRA{N0f8}, cycle))
  resize(win, extent(win) .+ 50)
  # `ERROR_OUT_OF_DATE_KHR` is triggered on the second call.
  draw_and_present(cycle, 0.5)
  draw_and_present(cycle, 0.5)
  @test any(!iszero, collect(BGRA{N0f8}, cycle))

  close(wm, win)
end;
