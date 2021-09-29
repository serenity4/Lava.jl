struct RenderPass
    area::Vk.Rect2D
    samples::Int
end

RenderPass(area; samples = 1) = RenderPass(area, samples)
RenderPass(area::NTuple{4,<:Integer}; samples = 1) = RenderPass(Vk.Rect2D(Vk.Offset2D(area[1:2]...), Vk.Extent2D(area[3:4]...)), samples)
