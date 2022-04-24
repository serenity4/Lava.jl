using ImageMagick: clamp01nan

function save_test_render(filename, data, h::UInt; tmp = false, clamp = false)
  clamp && (data = clamp01nan.(data))
  filename = render_file(filename; tmp)
  ispath(filename) && rm(filename)
  save(filename, data')
  @test stat(filename).size > 0
  @test hash(data) == h
end

color = attachment(device; format = Vk.FORMAT_R16G16B16A16_SFLOAT, usage = Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_TRANSFER_DST_BIT | Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT, dims = (1920, 1080))
pcolor = PhysicalAttachment(color)

include("examples/rectangle.jl")
include("examples/texture_drawing.jl")
include("examples/multisampling.jl")
