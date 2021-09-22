"""
Describes data that an object needs to be drawn, but without having a pipeline created yet.
"""
struct ShaderDependencies
    vertex_buffer::Buffer
    index_buffer::Optional{Buffer}
    descriptor_sets::Vector{DescriptorSet}
end

function Vk.update_descriptor_sets(device, shader_dependencies::ShaderDependencies, resources)
    Vk.update_descriptor_sets(
        device,
        map(Base.Fix1(WriteDescriptorSet, shader_dependencies), resources),
        [],
    )
end

struct DescriptorInfo
    type::Vk.DescriptorType
    "1-based index into a descriptor set."
    set_index::Int
    binding::Int
end

struct Descriptor
    set::DescriptorSet
    info::DescriptorInfo
end

function collect_bindings(shaders)
    binding_sets = Dictionary{Int,Vector{Vk.DescriptorSetLayoutBinding}}()
    for shader ∈ shaders
        for info ∈ shader.descriptor_infos
            push!(get!(binding_sets, info.set_index, Vk.DescriptorSetLayoutBinding[]), Vk.DescriptorSetLayoutBinding(info.binding, info.type, shader.source.stage; descriptor_count=1))
        end
    end
    if !all(collect(keys(binding_sets)) .== 0:length(binding_sets) - 1)
        error("Invalid layout description (non-contiguous binding sets from 0) in $binding_sets.")
    end
    collect(values(binding_sets))
end

function create_descriptor_set_layouts(shaders)
    isempty(shaders) && return DescriptorSetLayout[]
    dev = device(first(shaders))
    map(collect_bindings(shaders)) do bindings
        info = Vk.DescriptorSetLayoutCreateInfo(bindings)
        handle = unwrap(create(DescriptorSetLayout, dev, info))
        layout = Dictionary{Vk.DescriptorType, Int}()
        for binding in bindings
            set!(layout, binding.descriptor_type, get(layout, binding.descriptor_type, 0) + binding.descriptor_count)
        end
        DescriptorSetLayout(handle, layout)
    end
end
