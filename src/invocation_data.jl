mutable struct DataBlock
  bytes::Vector{UInt8}
  # Byte index from which a block ID or a buffer device address can be read with 2 words.
  device_addresses::Vector{Int}
  # Byte index from which a descriptor index can be read with 1 word.
  descriptor_ids::Vector{Int}
  type::DataType
end
DataBlock(@nospecialize(T::DataType), bytes) = DataBlock(bytes, Int[], Int[], T)

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

function DataBlock(x, layout::LayoutStrategy)
  data = serialize(x, layout)
  block = DataBlock(typeof(x), data)
  annotate!(block, layout, x)
  block
end

function annotate!(block, layout, x::T, offset = 0) where {T}
  isprimitivetype(T) && return add_annotations!(block, x, offset)
  if isstructtype(T)
    for i in 1:fieldcount(T)
      field_offset = if isa(layout, VulkanLayout) || isa(layout, ShaderLayout)
        dataoffset(layout, spir_type(T, layout.tmap; fill_tmap = false), i)
      else
        dataoffset(layout, T, i)
      end
      annotate!(block, layout, getfield(x, i), offset + field_offset)
    end
  end
end

function add_annotations!(block, x::T, byte_offset) where {T}
  if T === DeviceAddress && is_generated_address(x)
    push!(block.device_addresses, 1 + byte_offset)
    true
  elseif T === DescriptorIndex
    push!(block.descriptor_ids, 1 + byte_offset)
    true
  else
    false
  end
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
  layout::VulkanLayout
end

function read_device_address(block::DataBlock, i::Int)
  byte_idx = block.device_addresses[i]
  bytes = @view block.bytes[byte_idx:byte_idx + 7]
  only(reinterpret(DeviceAddress, bytes))
end

function ProgramInvocationData(blocks, descriptors, logical_buffers, root, layout)
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
  ProgramInvocationData(blocks, descriptors, logical_buffers, root, postorder(g, root), layout)
end

function ProgramInvocationData(root::DataBlock, ctx::InvocationDataContext, layout)
  blocks = first.(sort(collect(ctx.blocks.d); by = last))
  descriptors = first.(sort(collect(ctx.descriptors.d); by = last))
  logical_buffers = first.(sort(collect(ctx.buffers.d); by = last))
  root = ctx.blocks.d[root]
  ProgramInvocationData(blocks, descriptors, logical_buffers, root, layout)
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

function device_address_block!(allocator::LinearAllocator, gdescs::GlobalDescriptors, materialized_resources, node_id::NodeID, data::ProgramInvocationData)
  addresses = Dictionary{DataBlock, DeviceAddress}()
  for i in data.postorder_traversal
    block = data.blocks[i]
    patched = copy(block)
    patch_descriptors!(patched, gdescs, data.descriptors, node_id)
    patch_pointers!(patched, data, addresses, materialized_resources)
    address = allocate_data!(allocator, patched.bytes, patched.type, data.layout)
    insert!(addresses, block, address)
  end
  DeviceAddressBlock(addresses[data.blocks[data.root]])
end

"""
    @invocation_data <program|programs> begin
      b1 = @block a
      b2 = @block B(@address(b1), @descriptor(texture))
      @block C(1, 2, @address(b2))
    end

Create a [`ProgramInvocationData`](@ref) out of `@block` annotations,
which represent [`DataBlock`](@ref)s. The program (or programs) in which
this data will be used is necessary to derive a correct serialization scheme
which respects the layout requirements of these programs.

Within the body of `@invocation_data`, three additional macros are allowed:
- `@block` to create a new block.
- `@address` to reference an existing block as a [`DeviceAddress`](@ref).
- `@descriptor` to reference a descriptor (texture, image, sampler...) via a computed index.

All descriptors passed on to `@descriptor` will be preserved as part of the program invocation data.
Multiple references to the same descriptor will reuse the same index.

The last value of the block must be a [`DataBlock`](@ref), e.g. obtained with `@block`, and will be set as
the root block for the program invocation data.
"""
macro invocation_data(progs, ex)
  Meta.isexpr(ex, :block) || (ex = Expr(:block, ex))
  @gensym ctx blk desc object block_index buffer_index layout
  transformed = postwalk(ex) do subex
    if Meta.isexpr(subex, :macrocall)
      ex = @trymatch string(subex.args[1]) begin
        "@block" => quote
          $blk = $DataBlock($(subex.args[3]), $layout)
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
    progs = $(esc(progs))
    $(esc(layout)) = isa(progs, Program) ? progs.layout : merge_program_layouts(progs)
    ans = $(esc(transformed))
    isa(ans, DataBlock) || error("A data block must be provided as the last instruction to @invocation_data. Such a block can be obtained with @block.")
    ProgramInvocationData(ans, $(esc(ctx)), $(esc(layout)))
  end
end
