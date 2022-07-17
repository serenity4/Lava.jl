using Lava
using BenchmarkTools: @btime

include(joinpath(pkgdir(Lava), "test", "examples.jl"))

empty!(device.fence_pool)
GC.gc()

prog = rectangle_program(device)
@btime (render(program_1($device, $vdata, $pcolor, $prog)); empty!($device.fence_pool))
@profview [(render(program_1(device, vdata, pcolor, prog)); empty!(device.fence_pool)) for _ in 1:500]
