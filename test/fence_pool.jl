function signal_fence(device, fence)
  cb = Lava.request_command_buffer(device)
  wait(Lava.submit(cb, SubmissionInfo(; signal_fence = fence)))
end

@testset "Fence pool" begin
  pool = Lava.FencePool(device)
  fence = Lava.fence(pool)
  (; available, completed, pending) = pool
  @test length(available) == 9
  @test pending == [fence]
  @test isempty(completed)
  @test Lava.compact!(pool) == 0
  @test length(available) == 9 && pending == [fence] && isempty(completed)
  @test Lava.fence_status(fence) == Vk.NOT_READY
  @test signal_fence(device, fence)
  @test Lava.fence_status(fence) == Vk.SUCCESS
  @test Lava.compact!(pool) == 1
  @test length(available) == 10 && isempty(pending) && isempty(completed)

  fence = Lava.fence(pool)
  @test signal_fence(device, fence)
  empty!(available)
  fence2 = Lava.fence(pool)
  @test fence == fence2
end;
