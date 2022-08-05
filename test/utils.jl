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
