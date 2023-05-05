function shader(device, ex::Expr, execution_model::SPIRV.ExecutionModel, options)
  f, args = @match ex begin
    :($f($(args...))) => (f, args)
  end

  argtypes, storage_classes, variable_decorations = shader_decorations(ex)

  interface = :($ShaderInterface($execution_model;
    storage_classes = $(copy(storage_classes)),
    variable_decorations = $(deepcopy(variable_decorations)),
    features = $device.spirv_features,
  ))
  !isnothing(options) && push!(interface.args[2].args, :(execution_options = $options))
  call = Expr(:call, f, Expr.(:(::), argtypes)...)
  shader(device, call, interface)
end

function shader_decorations(ex::Expr)
  f, args = @match ex begin
    :($f($(args...))) => (f, args)
  end

  argtypes = []
  storage_classes = SPIRV.StorageClass[]
  variable_decorations = Dictionary{Int,Decorations}()
  input_location = -1
  output_location = -1
  for (i, arg) in enumerate(args)
    @switch arg begin
      @case :(::$T::$C)
      sc, decs = @match C begin
        Expr(:curly, sc, decs...) => (get_storage_class(sc), collect(decs))
        C => (get_storage_class(C), [])
      end
      has_decorations = !isempty(decs)
      isnothing(sc) && throw(ArgumentError("Unknown storage class provided in $(repr(arg))"))
      push!(storage_classes, sc)
      if sc in (SPIRV.StorageClassInput, SPIRV.StorageClassOutput) && has_decorations
        # Look if there are any decorations which declare the argument as a built-in variable.
        # We assume that there must be only one such declaration at maximum.
        for (j, dec) in enumerate(decs)
          isa(dec, Symbol) || continue
          builtin = get_builtin(dec)
          isnothing(builtin) && throw(ArgumentError("Unknown built-in decoration $dec in $(repr(arg))"))
          get!(Decorations, variable_decorations, i).decorate!(SPIRV.DecorationBuiltIn, builtin)
          deleteat!(decs, j)
          other_builtins = filter(!isnothing ∘ get_builtin, decs)
          !isempty(other_builtins) && throw(ArgumentError("More than one built-in decoration provided: $(join([builtin; other_builtins], ", "))"))
        end
      end
      for dec in decs
        (name, args) = @match dec begin
          Expr(:macrocall, name, source, args...) => (Symbol(string(name)[2:end]), args)
          _ => error("Expected macrocall (e.g. `@DescriptorSet(1)`), got $dec")
        end
        concrete_dec = get_decoration(name)
        isnothing(concrete_dec) && throw(ArgumentError("Unknown decoration $name in $(repr(arg))"))
        get!(Decorations, variable_decorations, i).decorate!(concrete_dec, args...)
      end
      if sc in (SPIRV.StorageClassInput, SPIRV.StorageClassOutput) && (!has_decorations || begin
            list = variable_decorations[i]
            !has_decoration(list, SPIRV.DecorationBuiltIn) && !has_decoration(variable_decorations[i], SPIRV.DecorationLocation)
        end)
        location = sc == SPIRV.StorageClassInput ? (input_location += 1) : (output_location += 1)
        get!(Decorations, variable_decorations, i).decorate!(SPIRV.DecorationLocation, UInt32(location))
      end
      push!(argtypes, T)
      @case _
      error("Expected argument type to be in the form `::<Type>::<Class>` at location $i (got $(repr(arg)))")
    end
  end

  argtypes, storage_classes, variable_decorations
end

function get_enum_if_defined(dec, ::Type{T}) where {T}
  isa(dec, Symbol) || return nothing
  prop = Symbol(nameof(T), dec)
  isdefined(SPIRV, prop) || return nothing
  value = getproperty(SPIRV, prop)
  isa(value, T) || return nothing
  value::T
end

get_builtin(dec) = get_enum_if_defined(dec, SPIRV.BuiltIn)
get_storage_class(dec) = get_enum_if_defined(dec, SPIRV.StorageClass)
get_decoration(dec) = get_enum_if_defined(dec, SPIRV.Decoration)

function shader(device, ex::Expr, interface)
  args = SPIRV.get_signature(ex)
  quote
    isa($device, $Device) || throw(ArgumentError(string("`Device` expected as first argument, got a value of type `", typeof($device), '`')))
    spec = $ShaderSpec($(args...), $interface)
    source = $ShaderSource($device, spec)
    $Shader($device, source)
  end
end
