function Base.show(io::IO, rg::RenderGraph)
  print(io, typeof(rg), '(', length(rg.nodes), " nodes, ", length(rg.resources), " resources)")
end

function Base.show(io::IO, ::MIME"text/plain", rg::RenderGraph)
  print(io, nameof(RenderGraph), " with ", length(rg.nodes), " nodes and ", length(rg.resources), " resources")
end

function Base.show(io::IO, command::GraphicsCommand)
  print(io, typeof(command), '(')
  print(io, command.program)
  command.data !== nothing && print(io, ", ", length(command.data.blocks), " data blocks")
  print(io, ", ", command.targets)
  print(io, ", ", length(command.resource_dependencies), " resources")
  print(io, ')')
end

function Base.show(io::IO, targets::RenderTargets)
  print(io, typeof(targets), '(')
  if isempty(targets.color)
    print(io, "no color target")
  elseif length(targets.color) == 1
    print(io, "color = ", targets.color[1])
  elseif length(targets.color) > 1
    print(io, "color = [", join(targets.color, ", "), ']')
  end
  targets.depth !== nothing && print(io, ", ", "depth = ", targets.depth)
  targets.stencil !== nothing && print(io, ", ", "stencil = ", targets.stencil)
  print(io, ')')
end

function Base.show(io::IO, shader::Shader)
  print(io, typeof(shader), '(')
  print(io, shader.info.mi)
  print(io, ", ", shader.shader_module.vks)
  !isempty(shader.specialization_constants) && print(io, ", ", shader.specialization_constants)
  print(io, ')')
end

function Base.show(io::IO, data::ProgramInvocationData)
  print(io, typeof(data), '(', length(data.blocks), " blocks, ", length(data.descriptors), " descriptors, ", length(data.logical_buffers), " logical buffers)")
end

function Base.show(io::IO, frame::Frame)
  print(io, typeof(frame), '(', handle(frame.image), ", ", handle(frame.image_acquired), ')')
end
