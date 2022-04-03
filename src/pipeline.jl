# abstract type PipelineType end

# abstract type Compute <: PipelineType end
# abstract type Graphics <: PipelineType end
# abstract type RayTracing <: PipelineType end

@enum PipelineType::Int8 begin
  PIPELINE_TYPE_GRAPHICS
  PIPELINE_TYPE_COMPUTE
  PIPELINE_TYPE_ASYNC_COMPUTE
end

function Vk.PipelineBindPoint(type::PipelineType)
  @match type begin
    &PIPELINE_TYPE_GRAPHICS => Vk.PIPELINE_BIND_POINT_GRAPHICS
    &(PIPELINE_TYPE_COMPUTE | PIPELINE_TYPE_ASYNC_COMPUTE) => Vk.PIPELINE_BIND_POINT_COMPUTE
  end
end

function PipelineType(bind_point::Vk.PipelineBindPoint)
  @match bind_point begin
    &Vk.PIPELINE_BIND_POINT_GRAPHICS => PIPELINE_TYPE_GRAPHICS
    &Vk.PIPELINE_BIND_POINT_COMPUTE => PIPELINE_TYPE_COMPUTE
  end
end

struct PipelineLayout <: LavaAbstraction
  handle::Vk.PipelineLayout
  descriptor_set_layouts::Vector{Vk.DescriptorSetLayout}
  push_constant_ranges::Vector{Vk.PushConstantRange}
end

struct Pipeline <: LavaAbstraction
  handle::Vk.Pipeline
  type::PipelineType
  layout::PipelineLayout
end

struct DepthBias
  constant_factor::Float32
  "Clamp limit. Negative values define a lower bound, while positive values define an upper bound."
  clamp::Float32
  slope_factor::Float32
end

struct PipelineState
  depth_bias::Optional{DepthBias}
  blending_mode
  enable_depth_testing::Bool
  enable_depth_clamp::Bool
  cull_mode::Vk.CullModeFlag
  triangle_orientation::Vk.FrontFace
  primitive_topology::Vk.PrimitiveTopology
  viewports::Vector{Vk.Viewport}
  scissors::Vector{Vk.Rect2D}
  logic_op::Vk.LogicOp
  depth_compare_op::Vk.CompareOp
  stencil_op::Vk.StencilOp
  enable_primitive_restart::Bool
  color_write::Vk.ColorComponentFlag
end

# PipelineState(;
#   depth_bias::Optional{DepthBias} = nothing,
#   blending_mode = nothing,
#   enable_depth_testing::Bool = false,
#   enable_depth_clamp::Bool = false,
#   cull_mode::Vk.CullModeFlag = Vk.CULL_MODE_BACK_BIT,
#   triangle_orientation::Vk.FrontFace = Vk.FRONT_FACE_CLOCKWISE,
#   primitive_topology::Vk.PrimitiveTopology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
#   viewports::Vector{Vk.Viewport},
#   scissors::Vector{Vk.Rect2D},
#   logic_op::Vk.LogicOp,
#   depth_compare_op::Vk.CompareOp,
#   stencil_op::Vk.StencilOp,
#   enable_primitive_restart::Bool = false,
#   color_write::Vk.ColorComponentFlag,
# )

"""
Pipeline-specific information.

This is different from draw or program state in that multiple `PipelineInfo` always require multiple pipelines.
"""
struct PipelineInfo
  # program::Program
  polygon_mode::Vk.PolygonMode
  multisample_state::Vk.PipelineMultisampleStateCreateInfo
  blend_enable::Bool
  blend_op::Vk.BlendOp
  descriptor_set_layouts::Vector{Vk.DescriptorSetLayout}
  push_constant_ranges::Vector{Vk.PushConstantRange}
  attachments::Vector{Vk.PipelineColorBlendAttachmentState}
  rendering_info::Vk.PipelineRenderingCreateInfo
end

function Vk.GraphicsPipelineCreateInfo(info::PipelineInfo) end
