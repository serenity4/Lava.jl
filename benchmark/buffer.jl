using Lava

using TimerOutputs
const to = TimerOutput()

instance, device = init(; with_validation = false, debug = false)

function f(device)
  @timeit to "Buffer creation" ibuffer = wait(buffer(device, collect(1:100)))
  @timeit to "Destroy buffer" begin
    finalize(ibuffer.handle)
    finalize(ibuffer.memory[].handle)
  end
end

@time f(device)

function manyf(f, args...)
  for _ in 1:1000
    f(args...)
  end
end

@time manyf(manual_f, device)
@profview manyf(f, device)
