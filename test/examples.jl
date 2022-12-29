function save_test_render(filename, data, h::Union{Nothing, UInt} = nothing; tmp = false, clamp = false)
  clamp && (data = clamp01nan.(data))
  filename = render_file(filename; tmp)
  ispath(filename) && rm(filename)
  save(filename, data')
  @test stat(filename).size > 0
  if !isnothing(h)
    @test hash(data) == h
  else
    hash(data)
  end
end

graphics_node(invocation = nothing) = RenderNode(render_area = RenderArea(1920, 1080), stages = Vk.PIPELINE_STAGE_2_VERTEX_SHADER_BIT | Vk.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, program_invocations = isnothing(invocation) ? ProgramInvocation[] : ProgramInvocation[invocation])

function render_graphics(device, node::RenderNode)
  rg = RenderGraph(device)
  add_node!(rg, node)
  render!(rg)
  invocation = node.program_invocations[end]
  color = only(invocation.targets.color)
  read_data(device, color)
end

read_data(device, color) = clamp01nan!(collect(RGBA{Float16}, color.data.view.image, device))

include("examples/textures.jl")

color = attachment_resource(device, nothing; format = Vk.FORMAT_R16G16B16A16_SFLOAT, usage_flags = Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_TRANSFER_DST_BIT | Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT, dims = [1920, 1080])

@testset "Examples" begin
  include("examples/rectangle.jl")
  include("examples/texture_drawing.jl")
  include("examples/multisampling.jl")
  include("examples/displacement.jl")
  include("examples/blur.jl")
end;
