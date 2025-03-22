include("examples/textures.jl")
include("examples/transforms.jl")

color = attachment_resource(device, nothing; format = RGBA{Float16}, usage_flags = Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_TRANSFER_DST_BIT | Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT, dims = [1920, 1080])
color_ms = attachment_resource(device, nothing; format = RGBA{Float16}, samples = 4, usage_flags = Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_TRANSFER_DST_BIT | Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT, dims = [1920, 1080])

@testset "Examples" begin
  include("examples/rectangle.jl")
  include("examples/texture_drawing.jl")
  include("examples/multisampling.jl")
  # XXX: Should use ShaderLibrary downstream tests instead, it's a fairly large maintenance burden otherwise.
  # include("examples/glyph.jl")
  # include("examples/text.jl")
  include("examples/blur.jl")
  # FIXME: Broken, vkCreateComputePipelines segfaults.
  # include("examples/boids.jl")
end;
