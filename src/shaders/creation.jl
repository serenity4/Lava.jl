function shader(device, ex::Expr, execution_model::SPIRV.ExecutionModel, options)
  f, args = @match ex begin
    :($f($(args...))) => (f, args)
  end

  argtypes = []
  storage_classes = SPIRV.StorageClass[]
  variable_decorations = Dictionary{Int,Decorations}()
  for (i, arg) in enumerate(args)
    @switch arg begin
      @case :(::$T::$C)
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
        get!(Decorations, variable_decorations, i).decorate!(SPIRV.DecorationLocation, count(i -> storage_classes[i] == sc && !haskey(variable_decorations, i), eachindex(storage_classes)))
      end
      push!(argtypes, T)
      @case _
      error("Expected argument type to be in the form `::<Type>::<Class>` at location $i (got $(repr(arg)))")
    end
  end

  interface = :($ShaderInterface($execution_model;
    storage_classes = $(copy(storage_classes)),
    variable_decorations = $(deepcopy(variable_decorations)),
    features = $device.spirv_features,
  ))
  !isnothing(options) && push!(interface.args[2].args, :(execution_options = $options))
  call = Expr(:call, f, Expr.(:(::), argtypes)...)
  shader(device, call, interface)
end

function shader(device, ex::Expr, interface)
  args = SPIRV.get_signature(ex)
  quote
    isa($device, $Device) || throw(ArgumentError(string("`Device` expected as first argument, got a value of type `", typeof($device), '`')))
    spec = $ShaderSpec($(args...), $interface)
    source = $ShaderSource($device, spec)
    $Shader($device, source)
  end
end
