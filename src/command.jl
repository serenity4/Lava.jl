@enum CommandType::UInt8 begin
  COMMAND_TYPE_DRAW_INDEXED = 1
  COMMAND_TYPE_DRAW_INDIRECT = 2
  COMMAND_TYPE_DRAW_INDEXED_INDIRECT = 3
  COMMAND_TYPE_DISPATCH = 10
  COMMAND_TYPE_DISPATCH_INDIRECT = 11
end

abstract type CommandImplementation end

struct Command
  type::CommandType
  impl::CommandImplementation
end

include("draw.jl")
include("dispatch.jl")

Command(draw::DrawIndexed) = Command(COMMAND_TYPE_DRAW_INDEXED, draw)
Command(draw::DrawIndirect) = Command(COMMAND_TYPE_DRAW_INDIRECT, draw)
Command(draw::DrawIndexedIndirect) = Command(COMMAND_TYPE_DRAW_INDEXED_INDIRECT, draw)
Command(dispatch::Dispatch) = Command(COMMAND_TYPE_DISPATCH, dispatch)
Command(dispatch::DispatchIndirect) = Command(COMMAND_TYPE_DISPATCH_INDIRECT, dispatch)
is_graphics_command(command::Command) = in(command.type, (COMMAND_TYPE_DRAW_INDEXED, COMMAND_TYPE_DRAW_INDIRECT, COMMAND_TYPE_DRAW_INDEXED_INDIRECT))
is_compute_command(command::Command) = in(command.type, (COMMAND_TYPE_DISPATCH, COMMAND_TYPE_DISPATCH_INDIRECT))
