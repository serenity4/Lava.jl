mutable struct DataBlock
  bytes::Vector{UInt8}
  # Byte index from which a block ID or a buffer device address can be read with 2 words.
  device_addresses::Vector{Int}
  # Byte index from which a descriptor index can be read with 1 word.
  descriptor_ids::Vector{Int}
  type::DataType
end
DataBlock(@nospecialize(T::DataType)) = DataBlock(UInt8[], Int[], Int[], T)

# We reserve the two most significant bits of a `DeviceAddress` in case we generate it via `@address`.
# The first bit indicates whether the address was generated (and therefore has its 2 MSBs which communicate the correct information).
# The second bit indicates whether the address is that of a logical buffer (1) or a data block (0).
is_generated_address(addr::DeviceAddress) = (UInt64(addr) >> 62) & 0x02 == 0x02
is_logical_buffer_address(addr::DeviceAddress) = (UInt64(addr) >> 62) & 0x03 == 0x03
is_block_address(addr::DeviceAddress) = (UInt64(addr) >> 62) & 0x03 == 0x02
extract_block_index(addr::DeviceAddress) = Int((UInt64(addr) << 2) >> 2)
extract_buffer_index(addr::DeviceAddress) = Int((UInt64(addr) << 2) >> 2)

generated_block_address(index) = DeviceAddress(index | (UInt64(0x02) << 62))
generated_logical_buffer_address(index) = DeviceAddress(index | (UInt64(0x03) << 62))

Base.copy(block::DataBlock) = @set block.bytes = copy(block.bytes)

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
  if T === DeviceAddress && is_generated_address(x)
    push!(block.device_addresses, lastindex(block.bytes) + 1)
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
  isempty(block.descriptor_ids) && isempty(block.device_addresses) && return @set block.bytes = align(block.bytes, t, type_info)

  aligned = DataBlock(UInt8[], Int[], Int[], block.type)

  remaps = Pair{UnitRange{Int},UnitRange{Int}}[]
  append!(aligned.bytes, align(block.bytes, t, type_info; callback = (from, to) -> push!(remaps, from => to)))

  i = firstindex(block.device_addresses)
  address_byte = i ≤ lastindex(block.device_addresses) ? block.device_addresses[i] : nothing
  j = firstindex(block.descriptor_ids)
  descriptor_byte = j ≤ lastindex(block.descriptor_ids) ? block.descriptor_ids[j] : nothing
  for (from, to) in remaps
    isnothing(address_byte) && isnothing(descriptor_byte) && break
    if address_byte === first(from)
      push!(aligned.device_addresses, first(to))
      address_byte = if i < lastindex(block.device_addresses)
        i += 1
        block.device_addresses[i]
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

mutable struct CounterDict{T}
  d::Dict{T,Int}
  counter::Int
end

CounterDict{T}() where {T} = CounterDict(Dict{T,Int}(), 0)

reserve!(cd::CounterDict, key) = get!(() -> (cd.counter += 1), cd.d, key)

struct InvocationDataContext
  blocks::CounterDict{DataBlock}
  descriptors::CounterDict{Descriptor}
  buffers::CounterDict{ResourceID}
end

InvocationDataContext() = InvocationDataContext(CounterDict{DataBlock}(), CounterDict{Descriptor}(), CounterDict{ResourceID}())

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
mutable struct ProgramInvocationData
  blocks::Vector{DataBlock}
  descriptors::Vector{Descriptor}
  logical_buffers::Vector{ResourceID}
  "Index of the block to use as interface."
  root::Int
  postorder_traversal::Vector{Int}
end
ProgramInvocationData() = ProgramInvocationData(DataBlock[], Descriptor[], ResourceID[], 0, Int[])

function read_device_address(block::DataBlock, i::Int)
  byte_idx = block.device_addresses[i]
  bytes = @view block.bytes[byte_idx:byte_idx + 7]
  only(reinterpret(DeviceAddress, bytes))
end

function ProgramInvocationData(blocks, descriptors, logical_buffers, root)
  g = SimpleDiGraph{Int}(length(blocks))
  for (i, block) in enumerate(blocks)
    for p in eachindex(block.device_addresses)
      addr = read_device_address(block, p)
      is_block_address(addr) || continue
      id = extract_block_index(addr)
      id < length(blocks) || error("Block id $(repr(id)) is used, but does not match any provided block.")
      add_edge!(g, i, id)
    end
  end
  ProgramInvocationData(blocks, descriptors, logical_buffers, root, postorder(g, root))
end

function ProgramInvocationData(root::DataBlock, ctx::InvocationDataContext)
  blocks = first.(sort(collect(ctx.blocks.d); by = last))
  descriptors = first.(sort(collect(ctx.descriptors.d); by = last))
  logical_buffers = first.(sort(collect(ctx.buffers.d); by = last))
  root = ctx.blocks.d[root]
  ProgramInvocationData(blocks, descriptors, logical_buffers, root)
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

function patch_block_reference(addr::DeviceAddress, data::ProgramInvocationData, addresses::Dictionary{DataBlock, DeviceAddress})
  index = extract_block_index(addr)
  pointee = data.blocks[index]
  @assert haskey(addresses, pointee) "Bad pointer dependency order detected."
  addresses[pointee]
end

function patch_buffer_reference(addr::DeviceAddress, data::ProgramInvocationData, resources)
  index = extract_buffer_index(addr)
  buffer_id = data.logical_buffers[index]
  buffer = get(resources, buffer_id, nothing)
  !isnothing(buffer) || error("Logical buffer $buffer_id is referenced in a data block, but is not known of the current cycle")
  @assert isphysical(buffer) && isbuffer(buffer)
  DeviceAddress(buffer)
end

function patch_pointers!(block::DataBlock, data::ProgramInvocationData, addresses::Dictionary{DataBlock, DeviceAddress}, resources)
  ptr = pointer(block.bytes)
  for i in eachindex(block.device_addresses)
    addr = read_device_address(block, i)
    patched_addr = if is_block_address(addr)
      patch_block_reference(addr, data, addresses)
    elseif is_logical_buffer_address(addr)
      patch_buffer_reference(addr, data, resources)
    else
      continue
    end
    unsafe_store!(Ptr{DeviceAddress}(ptr + block.device_addresses[i] - 1), patched_addr)
  end
end

function device_address_block!(allocator::LinearAllocator, gdescs::GlobalDescriptors, materialized_resources, node_id::NodeID, data::ProgramInvocationData, type_info::TypeInfo, layout::VulkanLayout)
  addresses = Dictionary{DataBlock, DeviceAddress}()
  for i in data.postorder_traversal
    block = data.blocks[i]
    aligned = align(block, type_info)
    patch_descriptors!(aligned, gdescs, data.descriptors, node_id)
    patch_pointers!(aligned, data, addresses, materialized_resources)
    address = allocate_data!(allocator, type_info, aligned.bytes, type_info.tmap[aligned.type], layout, false)
    insert!(addresses, block, address)
  end
  DeviceAddressBlock(addresses[data.blocks[data.root]])
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
  @gensym ctx blk desc object block_index buffer_index
  transformed = postwalk(ex) do subex
    if Meta.isexpr(subex, :macrocall)
      ex = @trymatch string(subex.args[1]) begin
        "@block" => quote
          $blk = $DataBlock($(subex.args[3]))
          $reserve!($ctx.blocks, $blk)
          $blk
        end
        "@address" => quote
          $object = $(subex.args[3])
          if isa($object, $Resource)
            $islogical($object) && $isbuffer($object) || error("Expected a logical buffer as resource in argument to `@address`, got ", $object)
            $buffer_index = $reserve!($ctx.buffers, $object.id)
            $generated_logical_buffer_address($buffer_index)
          elseif isa($object, $DataBlock)
            $block_index = $ctx.blocks.d[$object]
            $generated_block_address($block_index)
          else
            error("Expected a data block or logical buffer as argument to `@address`, got a `", typeof($object), '`')
          end
        end
        "@descriptor" => quote
          $desc = $(subex.args[3])
          isa($desc, $Descriptor) || error("Expected a `Descriptor` argument to `@descriptor`, got `", typeof($desc), '`')
          $DescriptorIndex($reserve!($ctx.descriptors, $desc))
        end
      end
      !isnothing(ex) && return ex
    end
    subex
  end
  quote
    $(esc(ctx)) = InvocationDataContext()
    ans = $(esc(transformed))
    isa(ans, DataBlock) || error("A data block must be provided as the last instruction to @invocation_data. Such a block can be obtained with @block.")
    ProgramInvocationData(ans, $(esc(ctx)))
  end
end
