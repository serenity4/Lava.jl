@precompile_all_calls begin
  @compile_traces verbose = false joinpath(@__DIR__, "precompilation_traces.jl")
end
