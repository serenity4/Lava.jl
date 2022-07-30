precompile_statements(statements_file, mods::Symbol...) = precompile_statements(statements_file, collect(mods))
function precompile_statements(statements_file, mods::Vector{Symbol})
  imports = [:(import $mod) for mod in mods]
  ex = quote
    using Base.Meta
    $(imports...)
    PrecompileStagingArea = Module()
    for (_pkgid, _mod) in Base.loaded_modules
        if !(_pkgid.name in ("Main", "Core", "Base"))
            eval(PrecompileStagingArea, :(const $(Symbol(_mod)) = $_mod))
        end
    end
    local n_succeeded = 0
    local n_failed = 0
    statements = collect(eachline($statements_file))
    print("Executing precompile statements...")
    for statement in statements
        try
            # println(statement)
            # This is taken from https://github.com/JuliaLang/julia/blob/2c9e051c460dd9700e6814c8e49cc1f119ed8b41/contrib/generate_precompile.jl#L375-L393
            ps = Meta.parse(statement)
            isexpr(ps, :call) || continue
            popfirst!(ps.args) # precompile(...)
            ps.head = :tuple
            l = ps.args[end]
            if (isexpr(l, :tuple) || isexpr(l, :curly)) && length(l.args) > 0 # Tuple{...} or (...)
                # XXX: precompile doesn't currently handle overloaded Vararg arguments very well.
                # Replacing N with a large number works around it.
                l = l.args[end]
                if isexpr(l, :curly) && length(l.args) == 2 && l.args[1] === :Vararg # Vararg{T}
                    push!(l.args, 100) # form Vararg{T, 100} instead
                end
            end
            # println(ps)
            ps = Core.eval(PrecompileStagingArea, ps)
            # XXX: precompile doesn't currently handle overloaded nospecialize arguments very well.
            # Skipping them avoids the warning.
            ms = length(ps) == 1 ? Base._methods_by_ftype(ps[1], 1, Base.get_world_counter()) : Base.methods(ps...)
            ms isa Vector || continue
            precompile(ps...)
            n_succeeded += 1
            print("\rExecuting precompile statements... $n_succeeded/$(length(statements))")
            if !iszero(n_failed)
              print(" (failed: ")
              printstyled(n_failed; bold = true, color = :red)
              print(')')
            end
          catch e
            # See julia issue #28808
            e isa InterruptException && rethrow()
            println()
            @warn "failed to execute $statement\n$(sprint(showerror, e))"
            n_failed += 1
        end
    end
    println()
    print("Successfully precompiled ")
    printstyled(n_succeeded; bold = true, color = :green)
    print(" statements")
    if !iszero(n_failed)
      print(" (")
      printstyled(n_failed; bold = true, color = :red)
      print(" failed)")
    end
    println()
  end

  # Exclude package precompilation from timings.
  Core.eval(Module(), quote $(imports...) end)

  @time Core.eval(Module(), ex)
end

# precompile_statements(:Lava)
precompile_statements(joinpath(@__DIR__, "statements", "precompile_statements.jl"), :Lava, :XCB, :ColorTypes, :OpenType, :GeometryExperiments, :ImageMagick, :ImageIO, :PNGFiles)
# precompile_statements("old_precompiles/statements/compiled2.jl", :Lava, :XCB, :ColorTypes, :OpenType, :GeometryExperiments, :ImageMagick, :ImageIO, :PNGFiles)
# precompile_statements("extra_precompile.jl", :Lava, :XCB, :ColorTypes, :OpenType, :GeometryExperiments, :ImageMagick, :ImageIO, :PNGFiles)
# precompile_statements("precompile_nomain.jl", :Lava, :XCB, :ColorTypes, :OpenType, :GeometryExperiments, :ImageMagick, :ImageIO, :PNGFiles)
