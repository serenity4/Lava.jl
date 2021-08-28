"""
Abstraction defined in the scope of this package.
"""
abstract type LavaAbstraction end

Base.unsafe_convert(T::Type{Ptr{Cvoid}}, x::LavaAbstraction) = Base.unsafe_convert(T, handle(x))

"""
Opaque handle to a foreign API data structure. Necessary to interact with external libraries such as Vulkan.
"""
function handle end

handle(x) = Vk.handle(x)

include("memory.jl")
include("buffer.jl")
include("image.jl")
include("command.jl")
include("pipeline.jl")
include("pool.jl")
include("shader.jl")
include("synchronization.jl")
