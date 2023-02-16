function signal_fence(device, fence)
  cb = request_command_buffer(device)
  wait(Lava.submit!(SubmissionInfo(; signal_fence = fence), cb))
end

@testset "Fence pool" begin
  pool = FencePool(device)
  fence = Lava.fence(pool)
  (; available, completed, pending) = pool
  @test length(available) == 9
  @test pending == [fence]
  @test isempty(completed)
  @test compact!(pool) == 0
  @test length(available) == 9 && pending == [fence] && isempty(completed)
  @test fence_status(fence) == Vk.NOT_READY
  @test signal_fence(device, fence)
  @test fence_status(fence) == Vk.SUCCESS
  @test compact!(pool) == 1
  @test length(available) == 10 && isempty(pending) && isempty(completed)

  fence = Lava.fence(pool)
  @test signal_fence(device, fence)
  empty!(available)
  fence2 = Lava.fence(pool)
  @test fence == fence2
end;
