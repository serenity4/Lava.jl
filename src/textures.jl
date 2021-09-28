"""
Sampling parameters to use with a texture.
"""
Base.@kwdef struct Sampling
    magnification::Vk.Filter = Vk.FILTER_CUBIC_IMG
    minification::Vk.Filter = Vk.FILTER_CUBIC_IMG
    mipmap_mode::Vk.SamplerMipmapMode = Vk.SAMPLER_MIPMAP_MODE_LINEAR
    address_modes::NTuple{3,Vk.SamplerAddressMode} = ntuple(Returns(Vk.SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER), 3)
    mip_lod_bias::Float32 = 0.
    anistropy_enable::Bool = true
    max_anisotropy::Float32 = 1000.
    compare_enable::Bool = false
    compare_op::Vk.CompareOp = Vk.COMPARE_OP_GREATER_OR_EQUAL
    lod_bounds::NTuple{2,Float32} = (0., 1000.)
    border_color::Vk.BorderColor = Vk.BORDER_COLOR_INT_OPAQUE_BLACK
    unnormalized_coordinates::Bool = false
end

function Vk.Sampler(sampling::Sampling)
    create_info = Vk.SamplerCreateInfo(
        sampling.magnification,
        sampling.minification,
        sampling.mipmap_mode,
        sampling.address_modes...,
        sampling.mip_lod_bias,
        sampling.anistropy_enable,
        sampling.max_anisotropy,
        sampling.compare_enable,
        sampling.compare_op,
        sampling.lod_bounds...,
        sampling.border_color,
        sampling.unnormalized_coordinates,
    )
    Vk.Sampler(device, create_info)
end

const DEFAULT_SAMPLING = Sampling()

"""
Texture identified by name with sampling parameters.

This texture is to be transformed into a texture index (to index into an array of sampled images or combined image-samplers depending on whether sampling parameters are provided) to be included as a material parameter in push constant data.
"""
struct Texture
    name::Symbol
    sampling::Union{Nothing,Sampling}
end

Texture(name::Symbol) = Texture(name, nothing)
