macro propagate_errors(ex)
    @assert Meta.isexpr(ex, :(=), 2)
    sym, ex = esc.(ex.args)
    quote
        ret = $ex
        if iserror(ret)
            return unwrap_error(ret)
        else
            $sym = unwrap(ret)
        end
    end
end


"""
    @forward MyType.prop method1, method2, ...

Extend the provided methods by forwarding the property `prop` of `MyType` instances.
This will give, for a given `method`:
```julia
method(x::MyType, args...; kwargs...) = method(x.prop, args...; kwargs...)
```

"""
macro forward(ex, fs)
    T, prop = @match ex begin
        :($T.$prop) => (T, prop)
        _ => error("Invalid expression $ex, expected <Type>.<prop>")
    end

    fs = @match fs begin
        :(($(fs...),)) => fs
        :($mod.$method) => [fs]
        ::Symbol => [fs]
        _ => error("Expected a method or a tuple of methods, got $fs")
    end

    defs = map(fs) do f
        esc(:($f(x::$T, args...; kwargs...) = $f(x.$prop, args...; kwargs...)))
    end

    Expr(:block, defs...)
end
