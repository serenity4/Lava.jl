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
@auto_hash_equals struct Program
  type::ProgramType
  data::Any
  type_info::TypeInfo
end

struct GraphicsProgram
  vertex_shader::Shader
  fragment_shader::Shader
  # ... other fields to be defined, such as geometry/tessellation and/or mesh shaders.
end

function Program(compute::Shader)
  compute.info.interface.execution_model == SPIRV.ExecutionModelGLCompute || error("A compute shader is required to form a compute program.")
  Program(PROGRAM_TYPE_COMPUTE, compute, compute.info.type_info)
end

Program(vertex_shader::Shader, fragment_shader::Shader) = Program(GraphicsProgram(vertex_shader, fragment_shader))
function Program(graphics::GraphicsProgram)
  type_info = retrieve_type_info((graphics.vertex_shader, graphics.fragment_shader))
  Program(PROGRAM_TYPE_GRAPHICS, graphics, type_info)
end

function retrieve_type_info(shaders)
  info = TypeInfo()
  for shader in shaders
    (; tmap, offsets, strides) = shader.info.type_info
    for (T, t) in pairs(tmap)
      existing = get(info.tmap, T, nothing)
      if !isnothing(existing) && existing ≠ t
        existing ≈ t || error("Julia type $T maps to different SPIR-V types: $existing and $t.")
      else
        info.tmap[T] = t
        existing = t
      end
      if isa(t, StructType)
        t_offsets = get(info.offsets, existing, nothing)
        shader_offsets = get(offsets, t, nothing)
        if !isnothing(shader_offsets)
          if isnothing(t_offsets)
            insert!(info.offsets, existing, shader_offsets)
          else
            t_offsets == shader_offsets || error("SPIR-V type $t possesses member offset decorations that are inconsistent across shaders.")
          end
        end
      elseif isa(t, ArrayType)
        t_stride = get(info.strides, existing, nothing)
        shader_stride = get(strides, t, nothing)
        if !isnothing(shader_stride)
          if isnothing(t_stride)
            insert!(info.strides, existing, shader_stride)
          else
            t_stride == shader_stride || error("SPIR-V type $t possesses an array stride decoration that is inconsistent across shaders.")
          end
        end
      end
    end
  end
  info
end

struct ResourceDependency
  type::ShaderResourceType
  access::MemoryAccess
  clear_value::Optional{NTuple{4,Float32}}
  samples::Int64
end
ResourceDependency(type, access; clear_value = nothing, samples = 1) = ResourceDependency(type, access, clear_value, samples)

function Base.merge(x::ResourceDependency, y::ResourceDependency)
  @assert x.id === y.id
  ResourceDependency(x.id, x.type | y.type, x.access | y.access)
end
