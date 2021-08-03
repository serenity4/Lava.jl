abstract type LavaAbstraction end

"""
Opaque handle to a foreign API data structure. Necessary to interact with external libraries such as Vulkan.
"""
function handle end

handle(x) = x.handle

# Vulkan specific

include("memory.jl")
include("buffer.jl")
include("image.jl")
