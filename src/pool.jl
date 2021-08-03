"""
Pool that manages the allocation and deallocation of resources of type `T`.
"""
abstract type Pool{T} <: LavaAbstraction end
