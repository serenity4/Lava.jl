struct StatefulRecording
  program::RefValue{Program}
  render_state::RefValue{RenderState}
  invocation_state::RefValue{ProgramInvocationState}
  data::RefValue{DrawData}
end

StatefulRecording() = StatefulRecording(Ref{Program}(), Ref(RenderState()), Ref(ProgramInvocationState()), Ref(DrawData()))

set_program(state::StatefulRecording, program::Program) = state.program[] = program
set_render_state(state::StatefulRecording, render_state::RenderState) = state.render_state[] = render_state
set_invocation_state(state::StatefulRecording, invocation_state::ProgramInvocationState) = state.invocation_state[] = invocation_state
set_data(state::StatefulRecording, data::DrawData) = state.data[] = data

function set_material(state::StatefulRecording, material::UInt64)
  data = state.data[]
  set_data(state, @set data.material_data = material)
  nothing
end

render_state(state::StatefulRecording) = state.render_state[]
invocation_state(state::StatefulRecording) = state.invocation_state[]

allocate_material(rg::RenderGraph, program::Program, data) = allocate_material(rg.device, rg.allocator, program, data)
allocate_material(record_state::StatefulRecording, rg::RenderGraph, data) = allocate_material(rg, record_state.program[], data)

allocate_vertex_data(rg::RenderGraph, program::Program, data) = allocate_vertex_data(rg.device, rg.allocator, program, data)
allocate_vertex_data(record_state::StatefulRecording, rg::RenderGraph, data) = allocate_vertex_data(rg, record_state.program[], data)

set_material(record_state::StatefulRecording, rg::RenderGraph, material) = set_material(record_state, allocate_material(record_state, rg, material))

function DrawInfo(record_state::StatefulRecording, rg::RenderGraph, vdata, idata, color...; depth = nothing, stencil = nothing, instances = 1:1)
  isdefined(record_state.program, 1) || error("A program must be set before drawing.")
  program = record_state.program[]
  state = DrawState(render_state(record_state), invocation_state(record_state), record_state.data[])
  DrawInfo(
    DrawIndexed(0, append!(rg.index_data, idata), instances),
    program,
    RenderTargets(color...; depth, stencil),
    DrawState(render_state(record_state), invocation_state(record_state), record_state.data[]),
  )
end
