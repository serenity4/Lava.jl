struct DepthBias
  constant_factor::Float32
  "Clamp limit. Negative values define a lower bound, while positive values define an upper bound."
  clamp::Float32
  slope_factor::Float32
end

Base.@kwdef struct RenderState
  blending_mode = nothing
  rasterizer_discard_enable::Bool = false
  enable_depth_testing::Bool = true
  enable_depth_bounds_testing::Bool = false
  depth_bounds::Tuple{Float32,Float32} = (typemin(Float32), typemax(Float32))
  enable_depth_write::Bool = true
  depth_bias::Optional{DepthBias} = nothing
  depth_compare_op::Vk.CompareOp = Vk.COMPARE_OP_LESS_OR_EQUAL
  enable_depth_clamping::Bool = false
  color_write_mask::Vk.ColorComponentFlag = Vk.COLOR_COMPONENT_R_BIT | Vk.COLOR_COMPONENT_G_BIT | Vk.COLOR_COMPONENT_B_BIT
  enable_stencil_testing::Bool = false
  stencil_front::Vk.StencilOpState = Vk.StencilOpState(Vk.STENCIL_OP_ZERO, Vk.STENCIL_OP_KEEP, Vk.STENCIL_OP_ZERO, Vk.COMPARE_OP_GREATER_OR_EQUAL, 0x00000000, 0xffffffff, 1)
  stencil_back::Vk.StencilOpState = stencil_front
  line_width::Float32 = 1f0
end

Vk.PipelineDepthStencilStateCreateInfo(state::RenderState) = Vk.PipelineDepthStencilStateCreateInfo(
    state.enable_depth_testing,
    state.enable_depth_write,
    state.depth_compare_op,
    state.enable_depth_bounds_testing,
    state.enable_stencil_testing,
    state.stencil_front,
    state.stencil_back,
    state.depth_bounds...,
  )

Base.@kwdef struct ProgramInvocationState
  cull_mode::Vk.CullModeFlag = Vk.CULL_MODE_BACK_BIT
  triangle_orientation::Vk.FrontFace = Vk.FRONT_FACE_CLOCKWISE
  enable_primitive_restart::Bool = false
  primitive_topology::Vk.PrimitiveTopology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
  vertex_input_rate::Vk.VertexInputRate = Vk.VERTEX_INPUT_RATE_VERTEX
  polygon_mode::Vk.PolygonMode = Vk.POLYGON_MODE_FILL
end

struct DrawState
  render_state::RenderState
  invocation_state::ProgramInvocationState
end

DrawState() = DrawState(RenderState(), ProgramInvocationState())
