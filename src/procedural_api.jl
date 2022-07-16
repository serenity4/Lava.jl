struct StatefulRecording
  program::RefValue{Program}
  render_state::RefValue{RenderState}
  invocation_state::RefValue{ProgramInvocationState}
  data::RefValue{DrawData}
end

StatefulRecording() = StatefulRecording(Ref{Program}(), Ref(RenderState()), Ref(ProgramInvocationState()), Ref(DrawData()))

set_program(rec::StatefulRecording, program::Program) = rec.program[] = program
set_render_state(rec::StatefulRecording, render_state::RenderState) = rec.render_state[] = render_state
set_invocation_state(rec::StatefulRecording, invocation_state::ProgramInvocationState) = rec.invocation_state[] = invocation_state
set_data(rec::StatefulRecording, data::DrawData) = rec.data[] = data

function set_material(rec::StatefulRecording, material::UInt64)
  data = rec.data[]
  set_data(rec, @set data.material_data = material)
  nothing
end

render_state(rec::StatefulRecording) = rec.render_state[]
invocation_state(rec::StatefulRecording) = rec.invocation_state[]

allocate_material(rg::RenderGraph, program::Program, data) = allocate_material(rg.device, rg.allocator, program, data)
allocate_material(rec::StatefulRecording, rg::RenderGraph, data) = allocate_material(rg, rec.program[], data)

allocate_vertex_data(rg::RenderGraph, program::Program, data) = allocate_vertex_data(rg.device, rg.allocator, program, data)
allocate_vertex_data(rec::StatefulRecording, rg::RenderGraph, data) = allocate_vertex_data(rg, rec.program[], data)

set_material(rec::StatefulRecording, rg::RenderGraph, material) = set_material(rec, allocate_material(rec, rg, material))

function DrawInfo(rec::StatefulRecording, rg::RenderGraph, vdata, idata, color...; depth = nothing, stencil = nothing, instances = 1:1)
  isdefined(rec.program, 1) || error("A program must be set before drawing.")
  program = rec.program[]
  data = rec.data[]
  @reset data.vertex_data = allocate_vertex_data(rg, program, vdata)
  set_data(rec, data)
  state = DrawState(render_state(rec), invocation_state(rec), data)
  DrawInfo(
    DrawIndexed(0, append!(rg.index_data, idata), instances),
    program,
    RenderTargets(color...; depth, stencil),
    DrawState(render_state(rec), invocation_state(rec), rec.data[]),
  )
end
