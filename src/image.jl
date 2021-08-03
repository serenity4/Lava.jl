"""
Image with dimension `N`.
"""
abstract type Image{N} <: LavaAbstraction end

"""
View of a resource, such as an image or buffer.
"""
abstract type View{O<:LavaAbstraction} end
