struct BakedRenderGraph
  global_data::GlobalData
  render_passes::Dictionary{UUID,Vk.RenderPass}
  image_layouts::Dictionary{UUID,Vk.ImageLayout}
  "Execution dependencies that must be preserved during execution."
  resources::PhysicalResources
end

function BakedRenderGraph(device, nodes)
  gd = GlobalData(device)
  records, pipeline_hashes = record_commands!(baked, nodes)
end

function bake(rg::RenderGraph, command_buffer)
  (; device) = rg
  nodes = sort_nodes(rg)
  baked = BakedRenderGraph(device, nodes)
  create_pipelines!(device)

  # fill command buffer with synchronization commands & recorded commands
  initialize(command_buffer, device, baked.global_data)
  flush(command_buffer, rg, nodes, records, pipeline_hashes)
  baked
end


function record_commands!(baked::BakedRenderGraph, nodes)
  records = CompactRecord[]
  pipeline_hashes = Dictionary{ProgramInstance,UInt64}()
  g = rg.resource_graph

  # record commands and submit pipelines for creation
  for node in nodes
    record = CompactRecord(baked.device, baked.global_data, node)
    node.render(record)
    push!(records, record)
    merge!(pipeline_hashes, submit_pipelines!(baked.device, baked, record))
  end

  records, pipeline_hashes
end

function Base.flush(cb::CommandBuffer, rg::RenderGraph, nodes, records, pipeline_hashes)
  binding_state = BindState()
  for (node, record) in zip(nodes, records)
    synchronize_before(cb, rg, node)
    begin_render_node(cb, rg, node)
    binding_state = flush(cb, record, device(rg), binding_state, pipeline_hashes)
    end_render_node(node, cb)
    synchronize_after(cb, rg, node)
  end
end

function end_render_node(cb, node)
  if !isnothing(node.render_pass)
    Vk.cmd_end_rendering(cb)
  end
end
