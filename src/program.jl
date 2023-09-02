@enum ProgramType::UInt8 begin
  PROGRAM_TYPE_COMPUTE = 1
  PROGRAM_TYPE_GRAPHICS = 2
  PROGRAM_TYPE_RAY_TRACING = 3
  # ... other types of programs can be defined as needed
end

"""
Computation unit that uses shaders as part of a graphics or compute pipeline.

It exposes a program interface through its shader interfaces and its shader resources.
"""
@struct_hash_equal struct Program
  type::ProgramType
  data::Any
  layout::VulkanLayout
end

struct GraphicsProgram
  vertex_shader::Shader
  fragment_shader::Shader
  # ... other fields to be defined, such as geometry/tessellation and/or mesh shaders.
end

function Program(compute::Shader)
  compute.info.interface.execution_model == SPIRV.ExecutionModelGLCompute || error("A compute shader is required to form a compute program.")
  Program(PROGRAM_TYPE_COMPUTE, compute, compute.info.layout)
end

Program(vertex_shader::Shader, fragment_shader::Shader) = Program(GraphicsProgram(vertex_shader, fragment_shader))
function Program(graphics::GraphicsProgram)
  layout = retrieve_layout((graphics.vertex_shader, graphics.fragment_shader))
  Program(PROGRAM_TYPE_GRAPHICS, graphics, layout)
end

function retrieve_layout(shaders)
  foldl((layout, shader) -> merge!(layout, shader.info.layout), shaders; init = VulkanLayout(shaders[1].info.layout.alignment))
end

function merge_program_layouts(progs)
  prog, progs... = progs
  foldl((layout, prog) -> merge!(layout, prog.layout), progs; init = VulkanLayout(prog.layout.alignment))
end
