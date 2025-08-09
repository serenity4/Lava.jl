texture_file(filename) = joinpath(@__DIR__, "resources", "textures", filename)
font_file(filename) = joinpath(@__DIR__, "resources", "fonts", filename)
render_file(filename; tmp = false) = joinpath(@__DIR__, "examples", "renders", tmp ? "tmp" : "", filename)

function test_validation_msg(f, test)
  val = Ref{Any}()
  mktemp() do path, io
    redirect_stderr(io) do
      val[] = f()
      yield()
    end
    seekstart(io)
    test(read(path, String))
  end
  val[]
end

function fake_graphics_command(
  targets::RenderTargets = RenderTargets();
  draw_state::DrawState = DrawState(),
  draw = DrawIndexed([1, 2, 3, 4]),
)
  graphics_command(draw, Program(PROGRAM_TYPE_GRAPHICS, nothing, VulkanLayout(VulkanAlignment())), DeviceAddressBlock(0), targets, draw_state)
end

fake_compute_command() = compute_command(Dispatch(1, 1, 1), Program(PROGRAM_TYPE_COMPUTE, nothing, VulkanLayout(VulkanAlignment())), DeviceAddressBlock(0))

read_transposed(::Type{T}, file) where {T} = permutedims(convert(Matrix{T}, load(file)), (2, 1))
read_transposed(file) = read_transposed(RGBA{Float16}, file)
read_png(::Type{T}, file) where {T} = read_transposed(T, file)
read_png(file) = read_png(RGBA{Float16}, file)
save_png(filename, data) = save(filename, PermutedDimsArray(data, (2, 1)))

function save_render(path, data)
  mkpath(dirname(path))
  ispath(path) && rm(path)
  save_png(path, data)
  path
end

"""
Save the rendered image `data` to `filename`, and test that the data is correct (by hash or approximately).

If no hash is provided, the render will just be saved. Otherwise, `data` will be saved at a temporary location, and a test is performed which depends on `h` and on the availability of a prior render at `filename`. The hashes used for the test are computed after reading the data back from the saved location, to ensure that the hash remains valid after writing to disk.

If a hash or a list of hashes is provided, the test will consist of checking that the computed hash equals or is contained in `h`.
Should this fail, `filename` will be checked for an existing render; if one is found, then a new test `data ≈ existing` is performed.
Should all these tests fail, a warning will be emitted with a link to a temporary file and to the reference file (if one exists).

If the existing render file has a different hash than the one that is provided, it will be overwritten only if the comparison test `data ≈ existing` fails, and the hash computed from `data` is correct.

If `keep = false` is provided, the provided render will be deleted after being created at a temporary location, even if it is not saved into `filename`.
"""
function save_test_render(filename, data::Matrix, h = nothing; keep = true)
  path = render_file(filename)
  path_tmp = tempname() * ".png"
  save_render(path_tmp, data)
  data = read_png(eltype(data), path_tmp)
  if isnothing(h)
    mv(path_tmp, path; force = true)
    return (path, hash(data))
  end
  @test stat(path_tmp).size > 0
  h′ = hash(data)
  (success, op_success, op_failure) = isa(h, UInt) ? (h′ == h, "==", "≠") : (h′ in h, "in", "∉")
  passes_with_data_comparison = false
  if isfile(path)
    existing = read_png(eltype(data), path)
    h′′ = hash(existing)
    existing_is_valid = h′′ == h || h′′ in h
    passes_with_data_comparison = isapprox(existing, data; rtol = 1e-2)
    if !success && existing_is_valid
      success |= passes_with_data_comparison
    end
  end
  if success
    if !passes_with_data_comparison
      isfile(path) && @assert !existing_is_valid
      keep && @info "$(isfile(path) ? "Updating" : "Creating") the render file at $path"
      mkpath(dirname(path))
      ispath(path) && rm(path)
      mv(path_tmp, path)
    end
  else
    msg = "Test failed: h′ $op_failure h ($(repr(h′)) $op_failure $(repr(h)))\nprovided image (h′) is available at $path_tmp"
    if isfile(path)
      h′′ = hash(existing)
      if isa(h, UInt) && existing_is_valid
        msg *= "\nreference image (h) is available at $path"
      else
        msg *= "\n\nThe reference image with hash $(repr(h′′)) is inconsistent with respect to the required hash$(isa(h, Integer) ? "" : "es") $(repr(h))"
      end
    end
    @warn "$msg"
  end
  if passes_with_data_comparison
    @test passes_with_data_comparison
  else
    isa(h, UInt) ? (@test h′ == h) : (@test h′ in h)
  end
  keep && return (path, h′)
  success && rm(path)
  h′
end

render_graphics(device, node::RenderNode) = render_graphics(device, node.commands[end])
render_graphics(device, command::Command) = render_graphics(device, only(command.graphics.targets.color), [command])
function render_graphics(device, result, nodes)
  render(device, nodes) || error("The computation did not terminate.")
  read_data(device, result)
end
read_data(device, attachment) = clamp01nan!(collect(attachment, device))
video_frame(frame::Matrix) = transpose(convert(Matrix{RGB{N0f8}}, frame))

"""
Remap a value from `(low1, high1)` to `(low2, high2)`.
"""
function remap(value, low1, high1, low2, high2)
  low2 + (value - low1) * (high2 - low2) / (high1 - low1)
end
