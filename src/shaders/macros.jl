macro vertex(device, ex, options = nothing)
  propagate_source(__source__, esc(shader(device, ex, SPIRV.ExecutionModelVertex, options)))
end

macro fragment(device, ex, options = nothing)
  propagate_source(__source__, esc(shader(device, ex, SPIRV.ExecutionModelFragment, options)))
end

macro compute(device, ex, options = nothing)
  propagate_source(__source__, esc(shader(device, ex, SPIRV.ExecutionModelGLCompute, options)))
end
