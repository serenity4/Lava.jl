"""
Computation unit that uses shaders as part of a graphics or compute pipeline.

It exposes a program interface through its shader interfaces and its shader resources.
"""
@auto_hash_equals struct Program
  shaders::Dictionary{Vk.ShaderStageFlag,Shader}
end

vertex_shader(prog::Program) = prog.shaders[Vk.SHADER_STAGE_VERTEX_BIT]
fragment_shader(prog::Program) = prog.shaders[Vk.SHADER_STAGE_FRAGMENT_BIT]

function Program(cache::ShaderCache, shaders...)
  shaders = map(shaders) do shader
    shader.stage => find_shader!(cache, shader)
  end
  Program(dictionary(shaders))
end

function Program(device, shaders...)
  Program(device.shader_cache, shaders...)
end

function allocate_pointer_resource!(allocator::LinearAllocator, data::AbstractVector, shader::Shader, layout::VulkanLayout)
  T = eltype(data)
  t = shader.source.typerefs[T]
  offsets = shader.source.offsets[t]
  # TODO: Check that the SPIR-V type of the load instruction corresponds to the type of `data`.
  # TODO: Get alignment from the extra operand MemoryAccessAligned of the corresponding OpLoad instruction.
  # TODO: This must be consistent with the stride of the loaded array.
  load_alignment = data_alignment(layout, t)
  start = get_offset(allocator, load_alignment)
  for el in data
    bytes = extract_bytes(el)
    isa(t, StructType) && (bytes = align(bytes, t, shader.source.offsets[t]))
    copyto!(allocator, bytes, load_alignment)
  end
  sub = @view allocator.buffer[start:allocator.last_offset]
  device_address(sub)
end

function allocate_pointer_resource!(allocator::LinearAllocator, data::T, shader::Shader, layout::VulkanLayout) where {T}
  bytes = extract_bytes(data)
  t = shader.source.typerefs[T]
  isa(t, StructType) && (bytes = align(bytes, t, shader.source.offsets[t]))
  # TODO: Check that the SPIR-V type of the load instruction corresponds to the type of `data`.
  # TODO: Get alignment from the extra operand MemoryAccessAligned of the corresponding OpLoad instruction.
  load_alignment = data_alignment(layout, t)
  sub = copyto!(allocator, bytes, load_alignment)
  device_address(sub)
end

data_alignment(layout::VulkanLayout, data) = data_alignment(layout, spir_type(typeof(data)))
data_alignment(layout::VulkanLayout, t::SPIRType) = alignment(layout, t, [SPIRV.StorageClassPhysicalStorageBuffer], false)

function allocate_vertex_data(allocator::LinearAllocator, program::Program, data, layout::VulkanLayout)
  # TODO: Make sure that the offsets and load alignment are consistent across all shaders that use this vertex data.
  allocate_pointer_resource!(allocator, data, vertex_shader(program), layout)
end

function allocate_vertex_data(device::Device, allocator::LinearAllocator, program::Program, data)
  allocate_vertex_data(allocator, program, data, device.layout)
end

function allocate_material(allocator::LinearAllocator, program::Program, data::T, layout::VulkanLayout) where {T}
  # TODO: Look up what shaders use the material and create pointer resource accordingly, instead of using this weird heuristic.
  shader = vertex_shader(program)
  !haskey(shader.source.typerefs, T) && (shader = fragment_shader(program))

  allocate_pointer_resource!(allocator, data, shader, layout)
end

function allocate_material(device_or_record::Device, allocator::LinearAllocator, program::Program, data)
  allocate_material(allocator, program, data, device_or_record.layout)
end

@auto_hash_equals struct RenderTargets
  color::Vector{PhysicalAttachment}
  depth::Optional{PhysicalAttachment}
  stencil::Optional{PhysicalAttachment}
end

RenderTargets(color::AbstractVector; depth = nothing, stencil = nothing) = RenderTargets(color, depth, stencil)
RenderTargets(color...; depth = nothing, stencil = nothing) = RenderTargets(collect(color); depth, stencil)

"""
Program to be compiled into a pipeline with a specific state.
"""
@auto_hash_equals struct ProgramInstance
  program::Program
  state::DrawState
  targets::RenderTargets
end
