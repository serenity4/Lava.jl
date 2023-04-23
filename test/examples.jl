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

render_graphics(device, node::RenderNode) = render_graphics(device, node.commands[end])
render_graphics(device, command::Command) = render_graphics(device, only(command.graphics.targets.color), [command])
function render_graphics(device, color, nodes)
  render(device, nodes)
  read_data(device, color)
end
read_data(device, color) = clamp01nan!(collect(RGBA{Float16}, color.attachment.view.image, device))
video_frame(frame::Matrix) = transpose(convert(Matrix{RGB{N0f8}}, frame))

include("examples/utils.jl")
include("examples/textures.jl")
include("examples/transforms.jl")

color = attachment_resource(device, nothing; format = Vk.FORMAT_R16G16B16A16_SFLOAT, usage_flags = Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_TRANSFER_DST_BIT | Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT, dims = [1920, 1080])
color_ms = attachment_resource(device, nothing; format = Vk.FORMAT_R16G16B16A16_SFLOAT, samples = 4, usage_flags = Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_TRANSFER_DST_BIT | Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT, dims = [1920, 1080])

@testset "Examples" begin
  include("examples/rectangle.jl")
  include("examples/texture_drawing.jl")
  include("examples/multisampling.jl")
  include("examples/glyph.jl")
  include("examples/displacement.jl")
  include("examples/blur.jl")
  include("examples/boids.jl")
end;
