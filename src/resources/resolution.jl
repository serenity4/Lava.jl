Dictionaries.index(rec::CompactRecord, texture::Texture) = index!(rec.gd.resources, texture, rec.image_layouts[(uuid(texture.image))])
Dictionaries.index(rec::CompactRecord, sampling::Sampling) = index!(rec.gd.resources, sampling)

function index!(descriptors::ResourceDescriptors, texture::Texture, image_layout::Vk.ImageLayout)::UInt32
  sampler = if !isnothing(texture.sampling)
    # Combined image sampler.
    Vk.Sampler(descriptors.pool.device, texture.sampling)
  else
    # Sampled image (sampler must be explicited in the shader).
    empty_handle(Vk.Sampler)
  end
  image_index!(descriptors, sampler, texture.view, image_layout)
end

function index!(descriptors::ResourceDescriptors, sampling::Sampling)
  view = empty_handle(Vk.ImageView)
  sampler = Vk.Sampler(descriptors.pool.device, sampling)
  sampler_index!(descriptors, sampler, view)
end
