shader_file(filename) = joinpath(@__DIR__, "resources", "shaders", filename)
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

function fake_graphics_command(;
  targets::Union{RenderTargets,Nothing} = nothing,
  draw_state::DrawState = DrawState(),
  draw = DrawIndexed([1, 2, 3, 4]),
)
  graphics_command(draw, Program(PROGRAM_TYPE_GRAPHICS, nothing, VulkanLayout(VulkanAlignment())), DeviceAddressBlock(0), targets, draw_state)
end

fake_compute_command() = compute_command(Dispatch(1, 1, 1), Program(PROGRAM_TYPE_COMPUTE, nothing, VulkanLayout(VulkanAlignment())), DeviceAddressBlock(0))
