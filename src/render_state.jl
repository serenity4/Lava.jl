Base.@kwdef struct RenderState
  blending_mode = nothing
  enable_depth_testing::Bool = false
  depth_bias::Optional{DepthBias} = nothing
  enable_depth_clamp::Bool = false
  color_write_mask::Vk.ColorComponentFlag = Vk.COLOR_COMPONENT_R_BIT | Vk.COLOR_COMPONENT_G_BIT | Vk.COLOR_COMPONENT_B_BIT
end

Base.@kwdef struct ProgramInvocationState
  cull_mode::Vk.CullModeFlag = Vk.CULL_MODE_BACK_BIT
  triangle_orientation::Vk.FrontFace = Vk.FRONT_FACE_CLOCKWISE
  enable_primitive_restart::Bool = false
  primitive_topology::Vk.PrimitiveTopology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
  vertex_input_rate::Vk.VertexInputRate = Vk.VERTEX_INPUT_RATE_VERTEX
  polygon_mode::Vk.PolygonMode = Vk.POLYGON_MODE_FILL
end

"""
Interface structure holding a device address as its single field.

This structure is necessary until SPIRV.jl can work around the requirement of
having interface block types be composite types.
"""
struct DeviceAddress
  addr::UInt64
end

SPIRV.Pointer{T}(addr::DeviceAddress) where {T} = SPIRV.Pointer{T}(addr.addr)

struct DrawState
  render_state::RenderState
  program_state::ProgramInvocationState
  user_data::DeviceAddress
end

DrawState() = DrawState(RenderState(), ProgramInvocationState(), DeviceAddress(0))
