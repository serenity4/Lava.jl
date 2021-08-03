abstract type CommandRecord <: LavaAbstraction end

"""
Operation lazily executed, for example recorded in a [`CommandRecord`](@ref)
"""
abstract type LazyOperation <: LavaAbstraction end

"""
Copy operation from one source to a destination.
"""
abstract type Copy{S,D} <: LazyOperation end

"""
Set a property of type `P` to an object of type `O`.
"""
abstract type Set{P,O} <: LazyOperation end
