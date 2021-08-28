"""
Image with dimension `N`.
"""
abstract type Image{N} <: LavaAbstraction end

"""
View of a resource, such as an image or buffer.
"""
abstract type View{O<:LavaAbstraction} end

Base.bind(image::Image, memory::Memory) = Vk.bind_image_memory(image, memory)
