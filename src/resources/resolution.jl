function texture_id!(resources, gd::GlobalData, device, rg::RenderGraph, tex::Texture, image_layout::Vk.ImageLayout)::UInt32
  view = View(resource_data(rg.resources[tex.image])::Image)
  sampler = if !isnothing(tex.sampling)
    # combined image sampler
    sampler = Vk.Sampler(device(rg), tex.sampling)
    # preserve sampler
    new!(resources, Resource(RESOURCE_CLASS_OTHER, sampler))
  else
    # sampled image
    empty_handle(Vk.Sampler)
  end
  add_image_descriptor!(gd, sampler, view, image_layout)
end

function sampler_id!(resources, gd::GlobalData, rg::RenderGraph, sampling::Sampling)
  view = empty_handle(Vk.ImageView)
  sampler = Vk.Sampler(device(rg), sampling)
  # preserve sampler
  new!(resources, Resource(RESOURCE_CLASS_OTHER, sampler))
  register(frame, gensym(:sampler), sampler; persistent = false)
  add_sampler_descriptor!(gd, sampler, view)
end
