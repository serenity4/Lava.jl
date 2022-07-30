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

primitive type DeviceAddress 64 end

DeviceAddress(address::UInt64) = reinterpret(DeviceAddress, address)

Base.convert(::Type{UInt64}, address::DeviceAddress) = reinterpret(UInt64, address)

SPIRV.primitive_type_to_spirv(::Type{DeviceAddress}) = SPIRV.IntegerType(64, 0)

const BlockUUID = UUID

mutable struct DataBlock
  bytes::Vector{UInt8}
  # Byte index from which a block ID can be read with 2 words.
  pointer_addresses::Vector{Int}
  # Byte index from which a descriptor index can be read with 1 word.
  descriptor_ids::Vector{Int}
  type::DataType
end
DataBlock(@nospecialize(T::DataType)) = DataBlock(UInt8[], Int[], Int[], T)

DeviceAddress(block::DataBlock) = DeviceAddress(objectid(block))

function DataBlock(x)
  block = DataBlock(typeof(x))
  extract!(block, x)
  block
end

extract!(block::DataBlock, x::T) where {T} = isstructtype(T) ? extract_struct!(block, x) : extract_leaf!(block, x)

function extract!(block::DataBlock, x::AbstractVector)
  for el in x
    extract!(block, el)
  end
end

function extract_struct!(block::DataBlock, x::T) where {T}
  for field in fieldnames(T)
    extract!(block, getfield(x, field))
  end
end

function extract_leaf!(block::DataBlock, x::T) where {T}
  if T === DeviceAddress
    push!(block.pointer_addresses, lastindex(block.bytes) + 1)
  elseif T === DescriptorIndex
    push!(block.descriptor_ids, lastindex(block.bytes) + 1)
  end
  append!(block.bytes, SPIRV.extract_bytes(x))
end

"""
Align a data block according to the type layout information provided by `type_info`.
"""
function SPIRV.align(block::DataBlock, type_info::TypeInfo)
  t = type_info.mapping[block.type]
  sizes = payload_sizes(t)
  offsets = getoffsets(type_info, t)
  res = DataBlock(block.type)
  append!(res.bytes, align(block.bytes, sizes, offsets))
  i = firstindex(block.pointer_addresses)
  address_byte = i ≤ lastindex(block.pointer_addresses) ? block.pointer_addresses[i] : nothing
  j = firstindex(block.descriptor_ids)
  descriptor_byte = j ≤ lastindex(block.descriptor_ids) ? block.descriptor_ids[j] : nothing
  data_byte = 1
  for (size, offset) in zip(sizes, offsets)
    isnothing(address_byte) && isnothing(descriptor_byte) && break
    if address_byte === data_byte
      push!(res.pointer_addresses, 1 + offset)
      address_byte = if i < lastindex(block.pointer_addresses)
        i += 1
        block.pointer_addresses[i]
      end
    end
    if descriptor_byte === data_byte
      push!(res.descriptor_ids, 1 + offset)
      descriptor_byte = if j < lastindex(block.descriptor_ids)
        j += 1
        block.descriptor_ids[j]
      end
    end
    data_byte += size
  end
  @assert isnothing(address_byte) && isnothing(descriptor_byte) "Addresses or descriptor indices were not transfered properly."
  res
end

const AllDescriptors = Union{Texture,LogicalImage,PhysicalImage,Sampling}

"""
Data attached to program invocations as a push constant.

There are three data transformations that must be done on this data before
being usable in a program:
- Descriptor patching with up to date descriptor indices.
- Memory padding with program-specific offsets.
- Pointer patching with new addresses at every cycle.

Memory padding cannot be done early, since different programs may have different member offset decorations.
Descriptor patching and memory padding can in theory be done in any order, before pointer patching. We choose
to do it after memory padding so that we can freely update in-place the newly padded bytes, which will use copies
of the original blocks.
"""
struct ProgramInvocationData
  blocks::Vector{DataBlock}
  "Index of the block to use as interface."
  root::Int
  descriptors::Vector{AllDescriptors}
  postorder_traversal::Vector{Int}
end

function ProgramInvocationData(blocks::AbstractVector{DataBlock}, descriptors, root)
  g = SimpleDiGraph{Int}(length(blocks))
  for (i, block) in enumerate(blocks)
    for byte_idx in block.pointer_addresses
      block_id = only(reinterpret(UInt64, @view block.bytes[byte_idx:byte_idx + 7]))
      j = findfirst(==(block_id) ∘ objectid, blocks)::Int
      add_edge!(g, i, j)
    end
  end
  ProgramInvocationData(blocks, root, descriptors, postorder(g, root))
end

function postorder(g, source)
  finish_times = zeros(Int, nv(g))
  postorder!(finish_times, g, [source], zeros(Bool, nv(g)), 0)
  sortperm(finish_times)
end

function postorder!(finish_times, g, next, visited, time)
  v = pop!(next)
  visited[v] = true
  for w in outneighbors(g, v)
    if !visited[w]
      push!(next, w)
      time = postorder!(finish_times, g, next, visited, time)
    end
  end
  finish_times[v] = (time += 1)
  time
end

function patch_descriptors!(block::DataBlock, ldescs::LogicalDescriptors, descriptors, node_id::NodeUUID)
  patched = Dictionary{UInt32,UInt32}()
  ptr = pointer(block.bytes)
  for byte_idx in block.descriptor_ids
    descriptor_idx = only(reinterpret(UInt32, @view block.bytes[byte_idx:byte_idx + 3]))
    index = get!(patched, descriptor_idx) do
      descriptor = descriptors[descriptor_idx]
      # TODO: This creates a new descriptor UUID.
      # We should first look through existing descriptors to reuse UUIDs.
      request_descriptor_index(ldescs, node_id, descriptor)
    end
    unsafe_store!(Ptr{UInt32}(ptr + byte_idx - 1), index)
  end
end

function patch_pointers!(block::DataBlock, addresses::Dictionary{UInt64, UInt64})
  ptr = pointer(block.bytes)
  for byte_idx in block.pointer_addresses
    block_id = only(reinterpret(UInt64, @view block.bytes[byte_idx:byte_idx + 7]))
    @assert haskey(addresses, block_id) "Bad pointer dependency order detected."
    unsafe_store!(Ptr{UInt64}(ptr + byte_idx - 1), addresses[block_id])
  end
end

function device_address_block!(allocator::LinearAllocator, ldescs::LogicalDescriptors, node_id::NodeUUID, data::ProgramInvocationData, type_info::TypeInfo, layout::VulkanLayout)
  addresses = Dictionary{UInt64, UInt64}()
  root_address = nothing
  for i in data.postorder_traversal
    block = data.blocks[i]
    aligned = align(block, type_info)
    patch_descriptors!(block, ldescs, data.descriptors, node_id)
    patch_pointers!(block, addresses)
    address = allocate_data!(allocator, type_info, aligned.bytes, type_info.mapping[aligned.type], layout, false)
    insert!(addresses, objectid(block), address)
    i == data.root && (root_address = address)
  end
  DeviceAddressBlock(root_address)
end

macro invocation_data(ex)
  Meta.isexpr(ex, :block) || error("Expected block expression, got expression of type ", ex.head)
  @gensym block_d descriptor_d block_counter descriptor_counter blk desc index
  transformed = postwalk(ex) do subex
    if Meta.isexpr(subex, :macrocall)
      ex = @trymatch string(subex.args[1]) begin
        "@block" => quote
          local $blk = $DataBlock($(subex.args[3]))
          $block_d[$blk] = ($block_counter += 1)
          $blk
        end
        "@address" => :($DeviceAddress($(subex.args[3])))
        "@descriptor" => quote
          local $desc = $(subex.args[3])
          local $index = ($descriptor_counter += 1)
          $descriptor_d[$desc] = $index
          $DescriptorIndex($index)
        end
      end
      !isnothing(ex) && return ex
    end
    subex
  end
  quote
    $(esc(block_d)) = IdDict{DataBlock,Int}()
    $(esc(descriptor_d)) = IdDict{AllDescriptors,Int}()
    $(esc(block_counter)) = 0
    $(esc(descriptor_counter)) = 0
    ans = $(esc(transformed))
    blocks = first.(sort(collect($(esc(block_d))); by = last))
    descriptors = first.(sort(collect($(esc(descriptor_d))); by = last))
    ProgramInvocationData(blocks, descriptors, $(esc(block_d))[ans])
  end
end

"""
Cycle-independent specification of a program invocation for graphics operations.
"""
struct ProgramInvocation
  program::Program
  # TODO: Generalize to other kinds of commands.
  command::DrawCommand
  targets::RenderTargets
  invocation_data::ProgramInvocationData
  render_state::RenderState
  invocation_state::ProgramInvocationState
end

function draw_info!(allocator::LinearAllocator, ldescs::LogicalDescriptors, program_invocation::ProgramInvocation, node_id::NodeUUID, device::Device)
  address_block = device_address_block!(allocator, ldescs, node_id, program_invocation.invocation_data, TypeInfo(program_invocation.program), device.layout)
  draw_state = DrawState(program_invocation.render_state, program_invocation.invocation_state, address_block)
  DrawInfo(program_invocation.command, program_invocation.program, program_invocation.targets, draw_state)
end

function allocate_array!(allocator::LinearAllocator, bytes::AbstractVector{UInt8}, elsize, eloffsets, load_alignment::Integer, elsizes = nothing)
  start = get_offset(allocator, load_alignment)
  for i in 0:(cld(length(bytes), elsize) - 1)
    elbytes = @view bytes[1 + i * elsize:(i + 1) * elsize]
    !isnothing(elsizes) && (elbytes = align(elbytes, elsizes, eloffsets))
    copyto!(allocator, elbytes, load_alignment)
  end
  sub = @view allocator.buffer[start:allocator.last_offset]
  device_address(sub)
end

function allocate_data!(allocator::LinearAllocator, type_info::TypeInfo, bytes::AbstractVector{UInt8}, type::SPIRV.ArrayType, layout::VulkanLayout)
  t = type.eltype
  # TODO: Check that the SPIR-V type of the load instruction corresponds to the type of `data`.
  # TODO: Get alignment from the extra operand MemoryAccessAligned of the corresponding OpLoad instruction.
  # TODO: This must be consistent with the stride of the loaded array.
  allocate_array!(allocator, bytes, payload_size(t), type_info.offsets[t], data_alignment(layout, t), payload_sizes(t))
end

"""
Allocate the provided bytes respecting the specified alignment.

The data must have been properly padded before hand with the correct shader offsets for it to be usable inside shaders.
"""
function allocate_data!(allocator::LinearAllocator, bytes::AbstractVector{UInt8}, load_alignment::Integer)
  sub = copyto!(allocator, bytes, load_alignment)
  device_address(sub)
end

"""
Allocate the provided bytes with an alignment computed from `layout`.

Padding will be applied to composite types using the offsets specified in the shader.
"""
function allocate_data!(allocator::LinearAllocator, type_info::TypeInfo, bytes::AbstractVector{UInt8}, t::SPIRType, layout::VulkanLayout, align_bytes = isa(t, StructType))
  align_bytes && (bytes = align(bytes, t, type_info.offsets[t]))
  # TODO: Check that the SPIR-V type of the load instruction corresponds to the type of `data`.
  # TODO: Get alignment from the extra operand MemoryAccessAligned of the corresponding OpLoad instruction.
  load_alignment = data_alignment(layout, t)
  allocate_data!(allocator, bytes, load_alignment)
end

data_alignment(layout::VulkanLayout, t::SPIRType) = alignment(layout, t, [SPIRV.StorageClassPhysicalStorageBuffer], false)

allocate_data!(allocator::LinearAllocator, data::T, type_info::TypeInfo, layout::VulkanLayout) where {T} = allocate_data!(allocator, type_info, extract_bytes(data), type_info.mapping[T], layout)
allocate_data!(allocator::LinearAllocator, data, shader::Shader, layout::VulkanLayout) where {T} = allocate_data!(allocator, data, shader.source.type_info, layout)

function allocate_data(allocator::LinearAllocator, program::Program, data::T, layout::VulkanLayout) where {T}
  # TODO: Look up what shaders use the data and create pointer resource accordingly, instead of using this weird heuristic.
  # TODO: Make sure that the offsets and load alignment are consistent across all shaders that use this data.
  shader = vertex_shader(program)
  !haskey(shader.source.type_info.mapping, T) && (shader = fragment_shader(program))
  allocate_data!(allocator, data, shader, layout)
end

"""
Program to be compiled into a pipeline with a specific state.
"""
@auto_hash_equals struct ProgramInstance
  program::Program
  state::DrawState
  targets::RenderTargets
end
