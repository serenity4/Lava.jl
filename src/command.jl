abstract type CommandRecord <: LavaAbstraction end

struct CommandBuffer <: CommandRecord
    command
end

"""
Operation lazily executed, for example recorded in a [`CommandRecord`](@ref)
"""
abstract type LazyOperation <: LavaAbstraction end

"""
Copy operation from one source to a destination.
"""
abstract type Copy{S,D} <: LazyOperation end

"""
Set a property of type `P` to an object of type `O`.
"""
abstract type SetProperty{P,O} <: LazyOperation end

abstract type DrawSource end

struct DrawIndirect{B<:Buffer} <: DrawSource
    draw_buffer::B
end

(draw::DrawIndirect)(record::CommandRecord, args...) = Vk.cmd_draw_indirect(record, draw.draw_buffer, args...)

struct DrawIndexed{B<:Buffer} <: DrawSource
    index_buffer::B
end

(draw::DrawIndexed)(record::CommandRecord, args...) = cmd_draw_indexed(record, args...)

struct DrawIndexedIndirect{B1<:Buffer,B2<:Buffer} <: DrawSource
    indirect::B1
    indexed::B2
end

(draw::DrawIndexedIndirect)(record::CommandRecord, args...) = cmd_draw_indexed_indirect(record, draw.indirect.buffer, args...)

struct DrawCommand{S<:DrawSource}
    info::DrawInfo
    call::S
end

function (draw::DrawCommand)(args...)
    bind(draw.info)
    draw.call(args...)
end

struct DrawInfo
    vbuffer::VertexBuffer
    ibuffer::Optional{IndexBuffer}
    descriptors::Optional{DescriptorSetVector}
    push_ranges::Vector{PushConstantRange}
    specialization_constants::Optional{SpecializationConstant}
end
