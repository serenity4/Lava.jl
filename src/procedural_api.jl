struct StatefulRecording
  program::RefValue{Program}
  render_state::RefValue{RenderState}
  invocation_state::RefValue{ProgramInvocationState}
  data_address::RefValue{DeviceAddress}
end

StatefulRecording() = StatefulRecording(Ref{Program}(), Ref(RenderState()), Ref(ProgramInvocationState()), Ref(DeviceAddress(0)))

set_program(rec::StatefulRecording, program::Program) = rec.program[] = program
set_render_state(rec::StatefulRecording, render_state::RenderState) = rec.render_state[] = render_state
set_invocation_state(rec::StatefulRecording, invocation_state::ProgramInvocationState) = rec.invocation_state[] = invocation_state
set_data(rec::StatefulRecording, addr::DeviceAddress) = rec.data_address[] = addr

render_state(rec::StatefulRecording) = rec.render_state[]
invocation_state(rec::StatefulRecording) = rec.invocation_state[]

set_data(rec::StatefulRecording, rg::RenderGraph, data) = set_data(rec, DeviceAddress(allocate_data(rg, rec.program[], data)))

allocate_data(rec::StatefulRecording, rg::RenderGraph, data) = allocate_data(rg, rec.program[], data)

function DrawInfo(rec::StatefulRecording, color...; depth = nothing, stencil = nothing, instances = 1:1)
  isdefined(rec.program, 1) || error("A program must be set before drawing.")
  DrawInfo(rec.program[], rec.data_address[], color...; depth, stencil, instances, invocation_state = invocation_state(rec), render_state = render_state(rec))
end
