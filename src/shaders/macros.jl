function shader(features, ex, execution_model)
  f, args = @match ex begin
    :($f($(args...))) => (f, args)
  end

  argtypes = []
  storage_classes = []
  variable_decorations = Dictionary{Int,Vector{Expr}}()
  for (i, arg) in enumerate(args)
    @switch arg begin
      @case :(::$C::$T)
      sc = C
      decs = []
      has_decorations = Meta.isexpr(sc, :curly)
      if has_decorations
        sc, decs... = sc.args
      end
      sc = getproperty(SPIRV, Symbol(:StorageClass, sc))::SPIRV.StorageClass
      push!(storage_classes, sc)
      if sc in (SPIRV.StorageClassInput, SPIRV.StorageClassOutput)
        for dec in decs
          push!(get!(Vector, variable_decorations, i), :(SPIRV.DecorationBuiltIn => [SPIRV.$(Symbol(:BuiltIn, dec))]))
        end
      else
        for dec in decs
          (dec_name, val) = @match dec begin
            :($d = $val) => (d, val)
            _ => error("Expected assignment, got $d")
          end
          push!(get!(Vector, variable_decorations, i), :(SPIRV.$(Symbol(:Decoration, dec_name)) => [$(esc(val))]))
        end
      end
      if !has_decorations && sc in (SPIRV.StorageClassInput, SPIRV.StorageClassOutput)
        push!(get!(Vector, variable_decorations, i), :(SPIRV.DecorationLocation => [$(UInt32(count(==(sc), storage_classes) - 1))]))
      end
      push!(argtypes, T)
      @case _
      error("Expected argument type to be in the form '::<Class>::<Type>")
    end
  end
  quote
    interface = SPIRV.ShaderInterface(
      execution_model = $execution_model,
      storage_classes = [$(storage_classes...)],
      variable_decorations = dictionary([
        $((:($i => dictionary([$(decs...)])) for (i, decs) in pairs(variable_decorations))...)
      ]),
      features = $(esc(features)),
    )
    $Lava.@shader interface $(esc(f))($((:(::$(esc(T))) for T in argtypes)...))
  end
end

macro fragment(features, ex)
  shader(features, ex, SPIRV.ExecutionModelFragment)
end

macro vertex(features, ex)
  shader(features, ex, SPIRV.ExecutionModelVertex)
end
