"""
Coordinate system for specifying image dimensions, such as width and height.
"""
abstract type SizeUnit end

"""
Size is given in pixels.
"""
struct SizeAbsolute <: SizeUnit end

"""
Size is given relative to the swapchain.
"""
struct SizeSwapchainRelative <: SizeUnit end

"""
Size is given relative to the viewport.
"""
struct SizeViewportRelative <: SizeUnit end
