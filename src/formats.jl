function format_type(format::Vk.Format)
  @match format begin
    &Vk.FORMAT_R8G8B8_UNORM => RGB{N0f8}
    &Vk.FORMAT_R8G8B8_SNORM => RGB{Q0f7}
    &Vk.FORMAT_B8G8R8_UNORM => BGR{N0f8}
    &Vk.FORMAT_B8G8R8_SNORM => BGR{Q0f7}
    &Vk.FORMAT_R8G8B8A8_UNORM => RGBA{N0f8}
    &Vk.FORMAT_R8G8B8A8_SNORM => RGBA{Q0f7}
    &Vk.FORMAT_B8G8R8A8_UNORM => BGRA{N0f8}
    &Vk.FORMAT_B8G8R8A8_SNORM => BGRA{Q0f7}
    &Vk.FORMAT_R16G16B16_UNORM => RGB{N0f16}
    &Vk.FORMAT_R16G16B16_SNORM => RGB{Q0f15}
    &Vk.FORMAT_R16G16B16_SFLOAT => RGB{Float16}
    &Vk.FORMAT_R16G16B16A16_UNORM => RGBA{N0f16}
    &Vk.FORMAT_R16G16B16A16_SNORM => RGBA{Q0f15}
    &Vk.FORMAT_R16G16B16A16_SFLOAT => RGBA{Float16}
    &Vk.FORMAT_R32G32B32_SFLOAT => RGB{Float32}
    &Vk.FORMAT_R32G32B32A32_SFLOAT => RGBA{Float32}
    &Vk.FORMAT_R64G64B64_SFLOAT => RGB{Float64}
    &Vk.FORMAT_R64G64B64A64_SFLOAT => RGBA{Float64}
    _ => nothing
  end
end

function format(@nospecialize(T::DataType))
  T <: AbstractArray && return format(eltype(T))
  @match T begin
    &RGB{N0f8} => Vk.FORMAT_R8G8B8_UNORM
    &RGB{Q0f7} => Vk.FORMAT_R8G8B8_SNORM
    &BGR{N0f8} => Vk.FORMAT_B8G8R8_UNORM
    &BGR{Q0f7} => Vk.FORMAT_B8G8R8_SNORM
    &RGBA{N0f8} => Vk.FORMAT_R8G8B8A8_UNORM
    &RGBA{Q0f7} => Vk.FORMAT_R8G8B8A8_SNORM
    &BGRA{N0f8} => Vk.FORMAT_B8G8R8A8_UNORM
    &BGRA{Q0f7} => Vk.FORMAT_B8G8R8A8_SNORM
    &RGB{N0f16} => Vk.FORMAT_R16G16B16_UNORM
    &RGB{Q0f15} => Vk.FORMAT_R16G16B16_SNORM
    &RGB{Float16} => Vk.FORMAT_R16G16B16_SFLOAT
    &RGBA{N0f16} => Vk.FORMAT_R16G16B16A16_UNORM
    &RGBA{Q0f15} => Vk.FORMAT_R16G16B16A16_SNORM
    &RGBA{Float16} => Vk.FORMAT_R16G16B16A16_SFLOAT
    &RGB{Float32} => Vk.FORMAT_R32G32B32_SFLOAT
    &RGBA{Float32} => Vk.FORMAT_R32G32B32A32_SFLOAT
    &RGB{Float64} => Vk.FORMAT_R64G64B64_SFLOAT
    &RGBA{Float64} => Vk.FORMAT_R64G64B64A64_SFLOAT
    _ => nothing
  end
end
