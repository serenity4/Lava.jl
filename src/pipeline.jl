@enum PipelineType::Int8 begin
  PIPELINE_TYPE_GRAPHICS
  PIPELINE_TYPE_COMPUTE
  PIPELINE_TYPE_ASYNC_COMPUTE
end

function Vk.PipelineBindPoint(type::PipelineType)
  @match type begin
    &PIPELINE_TYPE_GRAPHICS => Vk.PIPELINE_BIND_POINT_GRAPHICS
    &PIPELINE_TYPE_COMPUTE || &PIPELINE_TYPE_ASYNC_COMPUTE => Vk.PIPELINE_BIND_POINT_COMPUTE
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

struct RenderArea
  rect::Vk.Rect2D
end

RenderArea(x, y) = RenderArea(x, y, 0, 0)
function RenderArea(x, y, offset_x, offset_y)
  (x < 0 || y < 0) && throw(ArgumentError("A negative value was detected along `x` or `y` extents; all dimensions must be strictly positive."))
  (iszero(x) || iszero(y)) && throw(ArgumentError("A null value was detected along `x` or `y` extents; all dimensions must be strictly positive."))
  RenderArea(Vk.Rect2D(Vk.Offset2D(offset_x, offset_y), Vk.Extent2D(x, y)))
end

"""
Submit a pipeline create info for creation in the next batch.

A hash is returned to serve as the key to get the corresponding pipeline from the hash table.
"""
function pipeline_info_graphics(
  render_area::RenderArea,
  program::Program,
  state::RenderState,
  invocation_state::ProgramInvocationState,
  targets::RenderTargets,
  layout::PipelineLayout,
  resources,
)
  program_data = program.data::GraphicsProgram
  shader_stages = Vk.PipelineShaderStageCreateInfo.([program_data.vertex_shader, program_data.fragment_shader])
  # Vertex data is retrieved from an address provided in the push constant.
  vertex_input_state = Vk.PipelineVertexInputStateCreateInfo([], [])
  rendering_state = Vk.PipelineRenderingCreateInfo(targets)
  attachments = map(targets.color) do _
    if isnothing(state.blending_mode)
      #TODO: Allow specifying blending mode for color attachments.
      Vk.PipelineColorBlendAttachmentState(
        true,
        Vk.BLEND_FACTOR_SRC_ALPHA,
        Vk.BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        Vk.BLEND_OP_ADD,
        Vk.BLEND_FACTOR_SRC_ALPHA,
        Vk.BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        Vk.BLEND_OP_ADD;
        color_write_mask = state.color_write_mask,
      )
    else
      error("Color blending not supported")
    end
  end
  input_assembly_state = Vk.PipelineInputAssemblyStateCreateInfo(invocation_state.primitive_topology, false)
  (; x, y) = render_area.rect.offset
  (; width, height) = render_area.rect.extent
  viewport_state =
    Vk.PipelineViewportStateCreateInfo(
      viewports = [Vk.Viewport(x, y, float(width), float(height), 0, 1)],
      scissors = [render_area.rect],
    )

  (; depth_bias) = state
  use_depth_bias = !isnothing(depth_bias)
  depth_bias_constant_factor = depth_bias_clamp = depth_bias_slope_factor = 0.0f0
  if use_depth_bias
    depth_bias_constant_factor = depth_bias.constant_factor
    depth_bias_clamp = depth_bias.clamp
    depth_bias_slope_factor = depth_bias.slope
  end

  rasterizer = Vk.PipelineRasterizationStateCreateInfo(
    state.enable_depth_clamping,
    state.rasterizer_discard_enable,
    invocation_state.polygon_mode,
    invocation_state.triangle_orientation,
    use_depth_bias,
    depth_bias_constant_factor,
    depth_bias_clamp,
    depth_bias_slope_factor,
    state.line_width;
    invocation_state.cull_mode,
  )
  color_samples = samples.(get_physical_resource.(Ref(resources), targets.color))
  all(==(first(color_samples)), color_samples) || error("Incoherent number of samples detected for color attachments: ", color_samples)
  nsamples = first(color_samples)
  multisample_state = Vk.PipelineMultisampleStateCreateInfo(Vk.SampleCountFlag(nsamples), state.enable_fragment_supersampling, state.fragment_supersampling_rate, false, false)
  color_blend_state = Vk.PipelineColorBlendStateCreateInfo(false, Vk.LOGIC_OP_AND, attachments, ntuple(Returns(1.0f0), 4))
  depth_stencil_state = !isnothing(targets.depth) || !isnothing(targets.stencil) ? Vk.PipelineDepthStencilStateCreateInfo(state) : C_NULL
  dynamic_state = Vk.PipelineDynamicStateCreateInfo([Vk.DYNAMIC_STATE_DEPTH_TEST_ENABLE, Vk.DYNAMIC_STATE_DEPTH_WRITE_ENABLE, Vk.DYNAMIC_STATE_DEPTH_COMPARE_OP, Vk.DYNAMIC_STATE_STENCIL_TEST_ENABLE, Vk.DYNAMIC_STATE_STENCIL_OP, Vk.DYNAMIC_STATE_STENCIL_COMPARE_MASK, Vk.DYNAMIC_STATE_STENCIL_WRITE_MASK, Vk.DYNAMIC_STATE_STENCIL_REFERENCE])
  Vk.GraphicsPipelineCreateInfo(
    shader_stages,
    rasterizer,
    layout.handle,
    0, # subpass
    0; # base pipeline index
    next = rendering_state,
    input_assembly_state,
    vertex_input_state,
    viewport_state,
    multisample_state,
    color_blend_state,
    depth_stencil_state,
    dynamic_state,
  )
end

function pipeline_info_compute(program::Program, layout::PipelineLayout)
  shader = Vk.PipelineShaderStageCreateInfo(program.data::Shader)
  Vk.ComputePipelineCreateInfo(shader, layout.handle, 0)
end
