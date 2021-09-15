struct DescriptorSetLayout <: LavaAbstraction
    handle::Vk.DescriptorSetLayout
    layout::Dictionary{Vk.DescriptorType,Int}
end

vk_handle_type(::Type{DescriptorSetLayout}) = Vk.DescriptorSetLayout

mutable struct DescriptorPool <: LavaAbstraction
    handle::Vk.CommandPool
    allocated::Int
    size::Int
end

vk_handle_type(::Type{DescriptorPool}) = Vk.DescriptorPool

"""
Set of resources to be bound at an interface slot.
The higher the set, the less expensive rebindings will be.
"""
struct DescriptorSet <: LavaAbstraction
    handle::Vk.DescriptorSet
    layout::DescriptorSetLayout
end

struct DescriptorAllocator <: LavaAbstraction
    set_layout::Vk.DescriptorSetLayout
    pools::Vector{DescriptorPool}
    unused::Dictionary{DescriptorSet,Int}
    recycled::Vector{DescriptorSet}
    pool_size::Int
end

Base.broadcastable(da::DescriptorAllocator) = Ref(da)

function DescriptorAllocator(set_layout, size::Integer = 1000)
    DescriptorAllocator(pool, set_layout, size)
end

function allocate_pool(da::DescriptorAllocator)
    handle = Vk.DescriptorPool(device(da.set_layout), da.set_layout, da.pool_size, map(Base.splat(Vk.DescriptorPoolSize), pairs(set_layout.layout) .* da.pool_size))
    pool = DescriptorPool(handle, 0, da.pool_size)
    push!(da.pools, pool)
    pool
end

function find_pool!(da::DescriptorAllocator)
    idx = findfirst(x -> x.allocated < x.size, da.pools)
    if isnothing(idx)
        # allocate a new pool
        allocate_pool(da)
    else
        da.pools[idx]
    end
end

function allocate_descriptor_set(da::DescriptorAllocator)
    pool = find_pool!(da)
    allocate_info = Vk.DescriptorSetAllocateInfo(pool, [da.set_layout])
    set = first(unwrap(Vk.allocate_descriptor_sets(device(pool), allocate_info)))
    pool.allocated += 1
    DescriptorSet(set, da.set_layout)
end

function DescriptorSet(da::DescriptorAllocator)
    if !isempty(da.recycled)
        pop!(da.recycled)
    else
        allocate_descriptor_set(da)
    end
end
