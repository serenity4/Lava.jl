using Lava
using BenchmarkTools: @btime

include(joinpath(pkgdir(Lava), "test", "examples.jl"))

empty!(device.fence_pool)
GC.gc()

prog = rectangle_program(device)
vdata = [
  PosColor(Vec2(-0.5, 0.5), Arr{Float32}(1.0, 0.0, 0.0)),
  PosColor(Vec2(-0.5, -0.5), Arr{Float32}(0.0, 1.0, 0.0)),
  PosColor(Vec2(0.5, 0.5), Arr{Float32}(1.0, 1.0, 1.0)),
  PosColor(Vec2(0.5, -0.5), Arr{Float32}(0.0, 0.0, 1.0)),
]
@btime render(program_1($device, $vdata, $pcolor; prog = $prog))
@profview [(render(program_1(device, vdata, pcolor; prog))) for _ in 1:500]

prog = texture_program(device)
vdata = [
  TextureCoordinates(Vec2(-0.5, 0.5), Vec2(0.0, 0.0)),
  TextureCoordinates(Vec2(-0.5, -0.5), Vec2(0.0, 1.0)),
  TextureCoordinates(Vec2(0.5, 0.5), Vec2(1.0, 0.0)),
  TextureCoordinates(Vec2(0.5, -0.5), Vec2(1.0, 1.0)),
]
@btime (render(program_2($device, $vdata, $pcolor; prog = $prog)))
@profview [(render(program_2(device, vdata, pcolor; prog))) for _ in 1:500]
