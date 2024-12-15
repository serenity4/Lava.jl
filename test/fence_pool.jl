function signal_fence(device, fence)
  cb = request_command_buffer(device)
  wait(Lava.submit!(SubmissionInfo(; signal_fence = fence), cb))
end

@testset "Fence pool" begin
  pool = FencePool(device)

  fence = get_fence!(pool)
  (; available) = pool
  @test length(available) == 9
  @test available[begin] !== fence
  @test !is_signaled(fence)
  @test signal_fence(device, fence)
  @test is_signaled(fence)
  recycle!(fence)
  @test length(available) == 10
  @test available[end] === fence
  recycle!(fence)
  @test length(available) == 10

  @test fence === get_fence!(pool; signaled = true)
  @test length(available) == 9
  @test available[begin] !== fence
  @test is_signaled(fence)
  recycle!(fence)

  fence_unsignaled = get_fence!(pool; signaled = false)
  @test fence !== fence_unsignaled
  @test !is_signaled(fence_unsignaled)
  @test length(available) == 9
  @test available[begin] !== fence_unsignaled
  @test !is_signaled(fence_unsignaled)
  recycle!(fence_unsignaled)

  @test fence === get_fence!(pool; signaled = true)
  @test length(available) == 9
  fence_unsignaled_1 = get_fence!(pool; signaled = false)
  @test length(available) == 8
  fence_unsignaled_2 = get_fence!(pool; signaled = false)
  @test length(available) == 7
  fence_2 = get_fence!(pool; signaled = true)
  @test length(available) == 7
  recycle!(fence)
  recycle!(fence_2)
  recycle!(fence_unsignaled_1)
  recycle!(fence_unsignaled_2)
  @test length(available) == 11

  empty!(available)
  @test isempty(pool)
  fence = get_fence!(pool; signaled = true)
  @test length(available) == 0
  recycle!(fence)
  @test length(available) == 1
  fence = get_fence!(pool; signaled = false)
  @test length(available) == 0
  recycle!(fence)
  @test length(available) == 1

  @test !isempty(pool)
  empty!(pool)
  @test isempty(pool)
end;
