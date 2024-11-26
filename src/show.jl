function Base.show(io::IO, rg::RenderGraph)
  print(io, nameof(RenderGraph), '(', length(rg.nodes), " nodes, ", length(rg.resources), " resources)")
end

function Base.show(io::IO, ::MIME"text/plain", rg::RenderGraph)
  print(io, nameof(RenderGraph), " with ", length(rg.nodes), " nodes and ", length(rg.resources), " resources")
end

function Base.show(io::IO, rg::BakedRenderGraph)
  print(io, BakedRenderGraph, '(', length(rg.nodes), " nodes, ", length(rg.resources), " resources)")
end

function Base.show(io::IO, data::ProgramInvocationData)
  print(io, ProgramInvocationData, '(', length(data.blocks), " blocks, ", length(data.descriptors), " descriptors, ", length(data.logical_buffers), " logical buffers)")
end

function Base.show(io::IO, frame::Frame)
  print(io, Frame, '(', handle(frame.image), ", ", handle(frame.image_acquired), ')')
end
