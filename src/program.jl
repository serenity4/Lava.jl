"""
Computation unit that uses shaders as part of a graphics or compute pipeline.

It exposes a program interface through its shader interfaces and its shader resources.
"""
@auto_hash_equals struct Program
  shaders::Dictionary{Vk.ShaderStageFlag,Shader}
  type_info::TypeInfo
end

vertex_shader(prog::Program) = prog.shaders[Vk.SHADER_STAGE_VERTEX_BIT]
fragment_shader(prog::Program) = prog.shaders[Vk.SHADER_STAGE_FRAGMENT_BIT]

function Program(cache::ShaderCache, shaders...)
  type_info = retrieve_type_info(shaders)
  shaders = map(shaders) do shader
    shader.stage => find_shader!(cache, shader)
  end
  Program(dictionary(shaders), type_info)
end

function retrieve_type_info(shaders)
  info = TypeInfo()
  for shader in shaders
    (; tmap, offsets, strides) = shader.type_info
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

Program(device, shaders...) = Program(device.shader_cache, shaders...)

primitive type DeviceAddress 64 end

DeviceAddress(address::UInt64) = reinterpret(DeviceAddress, address)

Base.convert(::Type{UInt64}, address::DeviceAddress) = reinterpret(UInt64, address)
Base.convert(::Type{DeviceAddress}, address::UInt64) = reinterpret(DeviceAddress, address)

SPIRV.primitive_type_to_spirv(::Type{DeviceAddress}) = SPIRV.IntegerType(64, 0)
SPIRV.Pointer{T}(address::DeviceAddress) where {T} = Pointer{T}(convert(UInt64, address))

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

Base.copy(block::DataBlock) = @set block.bytes = copy(block.bytes)

DeviceAddress(block::DataBlock) = DeviceAddress(objectid(block))

function DataBlock(x)
  block = DataBlock(typeof(x))
  extract!(block, x)
  block
end

extract!(block::DataBlock, x::T) where {T} = isstructtype(T) ? extract_struct!(block, x) : extract_leaf!(block, x)

function extract!(block::DataBlock, x::Union{AbstractVector,Arr})
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
  t = type_info.tmap[block.type]
  isa(t, StructType) || isa(t, ArrayType) || return copy(block)
  isempty(block.descriptor_ids) && isempty(block.pointer_addresses) && return @set block.bytes = align(block.bytes, t, type_info)

  aligned = DataBlock(UInt8[], Int[], Int[], block.type)

  remaps = Pair{UnitRange{Int},UnitRange{Int}}[]
  append!(aligned.bytes, align(block.bytes, t, type_info; callback = (from, to) -> push!(remaps, from => to)))

  i = firstindex(block.pointer_addresses)
  address_byte = i ≤ lastindex(block.pointer_addresses) ? block.pointer_addresses[i] : nothing
  j = firstindex(block.descriptor_ids)
  descriptor_byte = j ≤ lastindex(block.descriptor_ids) ? block.descriptor_ids[j] : nothing
  for (from, to) in remaps
    isnothing(address_byte) && isnothing(descriptor_byte) && break
    if address_byte === first(from)
      push!(aligned.pointer_addresses, first(to))
      address_byte = if i < lastindex(block.pointer_addresses)
        i += 1
        block.pointer_addresses[i]
      end
    end
    if descriptor_byte === first(from)
      push!(aligned.descriptor_ids, first(to))
      descriptor_byte = if j < lastindex(block.descriptor_ids)
        j += 1
        block.descriptor_ids[j]
      end
    end
  end
  @assert isnothing(address_byte) && isnothing(descriptor_byte) "Addresses or descriptor indices were not transfered properly."
  aligned
end

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
  descriptors::Vector{Descriptor}
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

function patch_descriptors!(block::DataBlock, gdescs::GlobalDescriptors, descriptors, node_id::NodeID)
  patched = Dictionary{UInt32,UInt32}()
  ptr = pointer(block.bytes)
  for byte_idx in block.descriptor_ids
    descriptor_idx = only(reinterpret(UInt32, @view block.bytes[byte_idx:byte_idx + 3]))
    index = get!(patched, descriptor_idx) do
      descriptor = descriptors[descriptor_idx]
      @reset descriptor.node_id = node_id
      request_index!(gdescs, descriptor)
    end
    unsafe_store!(Ptr{DescriptorIndex}(ptr + byte_idx - 1), index)
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

function device_address_block!(allocator::LinearAllocator, gdescs::GlobalDescriptors, node_id::NodeID, data::ProgramInvocationData, type_info::TypeInfo, layout::VulkanLayout)
  addresses = Dictionary{UInt64, UInt64}()
  root_address = nothing
  for i in data.postorder_traversal
    block = data.blocks[i]
    aligned = align(block, type_info)
    patch_descriptors!(aligned, gdescs, data.descriptors, node_id)
    patch_pointers!(aligned, addresses)
    address = allocate_data!(allocator, type_info, aligned.bytes, type_info.tmap[aligned.type], layout, false)
    insert!(addresses, objectid(block), address)
    i == data.root && (root_address = address)
  end
  DeviceAddressBlock(root_address)
end

"""
    @invocation_data begin
      b1 = @block a
      b2 = @block B(@address(b1), @descriptor(texture))
      @block C(1, 2, @address(b2))
    end

Create a [`ProgramInvocationData`](@ref) out of `@block` annotations,
which represent [`DataBlock`](@ref)s.

Within the body of `@invocation_data`, three additional macros are allowed:
- `@block` to create a new block.
- `@address` to reference an existing block as a [`DeviceAddress`](@ref).
- `@descriptor` to reference a descriptor (texture, image, sampler...) via a computed index.

All descriptors passed on to `@descriptor` will be preserved as part of the program invocation data.
Multiple references to the same descriptor will reuse the same index.

The last value of the block must be a [`DataBlock`](@ref), e.g. obtained with `@block`, and will be set as
the root block for the program invocation data.
"""
macro invocation_data(ex)
  Meta.isexpr(ex, :block) || (ex = Expr(:block, ex))
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
          local $desc = $(subex.args[3])::$Descriptor
          local $index = get($descriptor_d, $desc, nothing)
          if isnothing($index)
            $index = ($descriptor_counter += 1)
            $descriptor_d[$desc] = $index
          end
          $DescriptorIndex($index)
        end
      end
      !isnothing(ex) && return ex
    end
    subex
  end
  quote
    $(esc(block_d)) = IdDict{DataBlock,Int}()
    $(esc(descriptor_d)) = IdDict{Descriptor,Int}()
    $(esc(block_counter)) = 0
    $(esc(descriptor_counter)) = 0
    ans = $(esc(transformed))::DataBlock
    blocks = first.(sort(collect($(esc(block_d))); by = last))
    descriptors = first.(sort(collect($(esc(descriptor_d))); by = last))
    ProgramInvocationData(blocks, descriptors, $(esc(block_d))[ans])
  end
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
  resource_dependencies::Dictionary{Resource, ResourceDependency}
end

function draw_info!(allocator::LinearAllocator, gdescs::GlobalDescriptors, program_invocation::ProgramInvocation, node_id::NodeID, device::Device)
  address_block = device_address_block!(allocator, gdescs, node_id, program_invocation.invocation_data, program_invocation.program.type_info, device.layout)
  draw_state = DrawState(program_invocation.render_state, program_invocation.invocation_state, address_block)
  DrawInfo(program_invocation.command, program_invocation.program, program_invocation.targets, draw_state)
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

Padding will be applied to composite and array types using the offsets specified in the shader.
"""
function allocate_data!(allocator::LinearAllocator, type_info::TypeInfo, bytes::AbstractVector{UInt8}, t::SPIRType, layout::VulkanLayout, align_bytes = isa(t, StructType) || isa(t, ArrayType))
  align_bytes && (bytes = align(bytes, t, type_info))
  # TODO: Check that the SPIR-V type of the load instruction corresponds to the type of `data`.
  # TODO: Get alignment from the extra operand MemoryAccessAligned of the corresponding OpLoad instruction.
  load_alignment = data_alignment(layout, t)
  allocate_data!(allocator, bytes, load_alignment)
end

data_alignment(layout::VulkanLayout, t::SPIRType) = alignment(layout, t, [SPIRV.StorageClassPhysicalStorageBuffer], false)

allocate_data!(allocator::LinearAllocator, data::T, type_info::TypeInfo, layout::VulkanLayout) where {T} = allocate_data!(allocator, type_info, extract_bytes(data), type_info.tmap[T], layout)
allocate_data!(allocator::LinearAllocator, data, shader::Shader, layout::VulkanLayout) = allocate_data!(allocator, data, shader.source.type_info, layout)

function allocate_data(allocator::LinearAllocator, program::Program, data::T, layout::VulkanLayout) where {T}
  # TODO: Look up what shaders use the data and create pointer resource accordingly, instead of using this weird heuristic.
  # TODO: Make sure that the offsets and load alignment are consistent across all shaders that use this data.
  shader = vertex_shader(program)
  !haskey(shader.source.type_info.tmap, T) && (shader = fragment_shader(program))
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
