struct Fence <: LavaAbstraction
  handle::Vk.Fence
  pool::Any
  Fence(handle, pool) = new(handle, convert(FencePool, pool))
end

vk_handle_type(::Type{Fence}) = Vk.Fence

function Base.getproperty(fence::Fence, name::Symbol)
  name === :pool && return getfield(fence, :pool)::FencePool
  getfield(fence, name)
end

status(fence::Fence) = unwrap(Vk.get_fence_status(fence.handle.device, fence.handle))
is_signaled(fence::Fence) = status(fence) == Vk.SUCCESS

Base.wait(fence::Fence, timeout = typemax(UInt64)) = wait_for_fences([fence.handle], timeout)
Base.wait(fences::AbstractVector{Fence}, timeout = typemax(UInt64)) = wait_for_fences([fence.handle for fence in fences], timeout)

function wait_for_fences(fences::AbstractVector{Vk.Fence}, timeout = typemax(UInt64))
  isempty(fences) && return true
  device = Vk.device(fences[1])
  ret = unwrap(Vk.wait_for_fences(device, fences, true, timeout))
  ret == Vk.SUCCESS
end

function Base.reset(fence::Fence)
  reset_fences([fence.handle])
  fence
end

Base.reset(fences::AbstractVector{Fence}) = reset_fences([fence.handle for fence in fences])

function reset_fences(fences::AbstractVector{Vk.Fence})
  isempty(fences) && return fences
  device = Vk.device(fences[1])
  unwrap(Vk.reset_fences(device, fences))
  fences
end

struct FencePool
  device::Vk.Device
  available::Vector{Fence}
end

function FencePool(device)
  pool = FencePool(device, Fence[])
  for i in 1:10
    push!(pool.available, Fence(pool))
  end
  pool
end

recycle!(fence::Fence) = recycle!(fence.pool, fence)
function recycle!(pool::FencePool, fence::Fence)
  in(fence, pool.available) && return pool
  push!(pool.available, fence)
  pool
end

"""
Retrieve a fence from the `pool`, creating a new one if none is already available.
"""
function get_fence!(pool::FencePool; signaled = false)
  (; device, available) = pool
  i = findfirst(==(signaled) ∘ is_signaled, available)
  if !isnothing(i)
    fence = available[i]
    deleteat!(available, i)
    return fence
  end
  signaled && return Fence(pool; signaled)
  i = findfirst(is_signaled, available)
  if !isnothing(i)
    fence = available[i]
    deleteat!(available, i)
    return reset(fence)
  end
  Fence(pool; signaled)
end

function Fence(pool::FencePool; signaled = false)
  flags = signaled ? Vk.FENCE_CREATE_SIGNALED_BIT : Vk.FenceCreateFlag()
  handle = Vk.Fence(pool.device; flags)
  Fence(handle, pool)
end

function Base.empty!(pool::FencePool)
  empty!(pool.available)
  pool
end

Base.isempty(pool::FencePool) = isempty(pool.available)

Base.show(io::IO, pool::FencePool) = print(io, FencePool, "(", pool.device, ", ", length(pool.available), " fences available)")
