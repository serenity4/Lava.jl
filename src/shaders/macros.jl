function shader(features, ex, execution_model)
  f, args = @match ex begin
    :($f($(args...))) => (f, args)
  end

  argtypes = []
  storage_classes = SPIRV.StorageClass[]
  variable_decorations = Dictionary{Int,Decorations}()
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
          builtin = getproperty(SPIRV, Symbol(:BuiltIn, dec))::SPIRV.BuiltIn
          get!(Decorations, variable_decorations, i).decorate!(SPIRV.DecorationBuiltIn, builtin)
        end
      else
        for dec in decs
          (dec_name, val) = @match dec begin
            :($d = $val) => (d, val)
            _ => error("Expected assignment, got $d")
          end
          concrete_dec = getproperty(SPIRV, Symbol(:Decoration, dec_name))::SPIRV.Decoration
          get!(Decorations, variable_decorations, i).decorate!(concrete_dec, val)
        end
      end
      if !has_decorations && sc in (SPIRV.StorageClassInput, SPIRV.StorageClassOutput)
        get!(Decorations, variable_decorations, i).decorate!(SPIRV.DecorationLocation, count(==(sc), storage_classes) - 1)
      end
      push!(argtypes, T)
      @case _
      error("Expected argument type to be in the form '::<Class>::<Type>")
    end
  end

  quote
    interface = SPIRV.ShaderInterface(
      execution_model = $execution_model,
      storage_classes = $(copy(storage_classes)),
      variable_decorations = $(deepcopy(variable_decorations)),
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
