Base.@kwdef struct LinearAllocatorPool
  device::Vk.Device
  allocators::Vector{LinearAllocator}
  n_allocated::Int64 = 10
  size_multiplier::Int64 = 2
end

function allocate!(pool::LinearAllocatorPool, delta::Vector{UInt8}, start = 1)
  for la in pool.allocators[start:end]
    if available_size(la) > length(delta)
      return copyto!(la, delta)
    end
  end
  alloc_size = size(last(pool.allocators))
  start = length(pool.allocators)
  append!(allocators, LinearAllocator(pool.device, alloc_size * pool.size_multiplier) for _ in 1:(pool.n_allocated))
  allocate!(pool, delta, start)
end

struct ResourceDeltas
  pool::LinearAllocatorPool
  deltas::Dictionary{UUID,SubBuffer{BufferBlock{MemoryBlock}}}
end

struct FrameTransition
  delta::ResourceDeltas
end

function (transition::FrameTransition)(prev::FrameGraph, next::FrameGraph)
  (; device) = prev
  @assert device === next.device
  prev_resources = physical_resources(prev)
  next_resources = physical_resources(next)
  for (name, resource) in pairs(transition.resources)
    name in keys(prev_resources) || name in keys(next_resources) || continue
    # Maybe can be added as a separate render pass?
    insert_barrier!(next, resource_barrier(name, prev_resources, next_resources))
    insert_update!(next, resource, resource.data)
  end
end
