function vertex_input_attribute_descriptions(::Type{T}, binding, formats=formats(T)) where {T}
    Vk.VertexInputAttributeDescription.(
        0:fieldcount(T)-1,
        binding,
        formats,
        fieldoffset.(T, 1:fieldcount(T)),
    )
end

formats(::Type{T}) where {T} = Vk.Format.(fieldtypes(T))

Vk.VertexInputBindingDescription(::Type{T}, binding; input_rate = VERTEX_INPUT_RATE_VERTEX) where {T} =
    Vk.VertexInputBindingDescription(binding, sizeof(T), input_rate)
