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
  append!(block.bytes, extract_bytes(x))
end

"""
Align a data block according to the type layout information provided by `type_info`.
"""
function SPIRV.align(block::DataBlock, type_info::TypeInfo)
  t = type_info.tmap[block.type]
  isa(t, StructType) || isa(t, ArrayType) || return copy(block)
  # Don't bother if we don't have to keep an external mapping for descriptors/pointers.
  # Perform the alignment and return the result.
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

function read_block_id(block::DataBlock, i::Int)
  byte_idx = block.pointer_addresses[i]
  bytes = @view block.bytes[byte_idx:byte_idx + 7]
  only(reinterpret(UInt64, bytes))
end

function ProgramInvocationData(blocks::AbstractVector{DataBlock}, descriptors, root)
  g = SimpleDiGraph{Int}(length(blocks))
  for (i, block) in enumerate(blocks)
    for p in eachindex(block.pointer_addresses)
      id = read_block_id(block, p)
      j = findfirst(==(id) ∘ objectid, blocks)
      isnothing(j) && error("Block id $(repr(id)) is used, but does not match any provided block.")
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
  for i in eachindex(block.pointer_addresses)
    id = read_block_id(block, i)
    @assert haskey(addresses, id) "Bad pointer dependency order detected."
    unsafe_store!(Ptr{UInt64}(ptr + 8(i - 1)), addresses[id])
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
    address = UInt64(allocate_data!(allocator, type_info, aligned.bytes, type_info.tmap[aligned.type], layout, false))
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
    ans = $(esc(transformed))
    isa(ans, DataBlock) || error("A data block must be provided as the last instruction to @invocation_data. Such a block can be obtained with @block.")
    blocks = first.(sort(collect($(esc(block_d))); by = last))
    descriptors = first.(sort(collect($(esc(descriptor_d))); by = last))
    ProgramInvocationData(blocks, descriptors, $(esc(block_d))[ans])
  end
end
