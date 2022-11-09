# Generated with formats_gen.jl

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

# Copy-paste from the Vulkan specification.

format(spirv_format::SPIRV.ImageFormat) = @match spirv_format begin
  &SPIRV.R8 => Vk.FORMAT_R8_UNORM
  &SPIRV.ImageFormatR8Snorm => Vk.FORMAT_R8_SNORM
  &SPIRV.ImageFormatR8ui => Vk.FORMAT_R8_UINT
  &SPIRV.ImageFormatR8i => Vk.FORMAT_R8_SINT
  &SPIRV.ImageFormatRg8 => Vk.FORMAT_R8G8_UNORM
  &SPIRV.ImageFormatRg8Snorm => Vk.FORMAT_R8G8_SNORM
  &SPIRV.ImageFormatRg8ui => Vk.FORMAT_R8G8_UINT
  &SPIRV.ImageFormatRg8i => Vk.FORMAT_R8G8_SINT
  &SPIRV.ImageFormatRgba8 => Vk.FORMAT_R8G8B8A8_UNORM
  &SPIRV.ImageFormatRgba8Snorm => Vk.FORMAT_R8G8B8A8_SNORM
  &SPIRV.ImageFormatRgba8ui => Vk.FORMAT_R8G8B8A8_UINT
  &SPIRV.ImageFormatRgba8i => Vk.FORMAT_R8G8B8A8_SINT
  &SPIRV.ImageFormatRgb10A2 => Vk.FORMAT_A2B10G10R10_UNORM_PACK32
  &SPIRV.ImageFormatRgb10a2ui => Vk.FORMAT_A2B10G10R10_UINT_PACK32
  &SPIRV.ImageFormatR16 => Vk.FORMAT_R16_UNORM
  &SPIRV.ImageFormatR16Snorm => Vk.FORMAT_R16_SNORM
  &SPIRV.ImageFormatR16ui => Vk.FORMAT_R16_UINT
  &SPIRV.ImageFormatR16i => Vk.FORMAT_R16_SINT
  &SPIRV.ImageFormatR16f => Vk.FORMAT_R16_SFLOAT
  &SPIRV.ImageFormatRg16 => Vk.FORMAT_R16G16_UNORM
  &SPIRV.ImageFormatRg16Snorm => Vk.FORMAT_R16G16_SNORM
  &SPIRV.ImageFormatRg16ui => Vk.FORMAT_R16G16_UINT
  &SPIRV.ImageFormatRg16i => Vk.FORMAT_R16G16_SINT
  &SPIRV.ImageFormatRg16f => Vk.FORMAT_R16G16_SFLOAT
  &SPIRV.ImageFormatRgba16 => Vk.FORMAT_R16G16B16A16_UNORM
  &SPIRV.ImageFormatRgba16Snorm => Vk.FORMAT_R16G16B16A16_SNORM
  &SPIRV.ImageFormatRgba16ui => Vk.FORMAT_R16G16B16A16_UINT
  &SPIRV.ImageFormatRgba16i => Vk.FORMAT_R16G16B16A16_SINT
  &SPIRV.ImageFormatRgba16f => Vk.FORMAT_R16G16B16A16_SFLOAT
  &SPIRV.ImageFormatR32ui => Vk.FORMAT_R32_UINT
  &SPIRV.ImageFormatR32i => Vk.FORMAT_R32_SINT
  &SPIRV.ImageFormatR32f => Vk.FORMAT_R32_SFLOAT
  &SPIRV.ImageFormatRg32ui => Vk.FORMAT_R32G32_UINT
  &SPIRV.ImageFormatRg32i => Vk.FORMAT_R32G32_SINT
  &SPIRV.ImageFormatRg32f => Vk.FORMAT_R32G32_SFLOAT
  &SPIRV.ImageFormatRgba32ui => Vk.FORMAT_R32G32B32A32_UINT
  &SPIRV.ImageFormatRgba32i => Vk.FORMAT_R32G32B32A32_SINT
  &SPIRV.ImageFormatRgba32f => Vk.FORMAT_R32G32B32A32_SFLOAT
  &SPIRV.ImageFormatR64ui => Vk.FORMAT_R64_UINT
  &SPIRV.ImageFormatR64i => Vk.FORMAT_R64_SINT
  &SPIRV.ImageFormatR11fG11fB10f => Vk.FORMAT_B10G11R11_UFLOAT_PACK32
  _ => error("Unknown SPIR-V image format $spirv_format")
end

spirv_format(format::Vk.Format) = @match format begin
  &Vk.FORMAT_R8_UNORM => SPIRV.ImageFormatR8
  &Vk.FORMAT_R8_SNORM => SPIRV.ImageFormatR8Snorm
  &Vk.FORMAT_R8_UINT => SPIRV.ImageFormatR8ui
  &Vk.FORMAT_R8_SINT => SPIRV.ImageFormatR8i
  &Vk.FORMAT_R8G8_UNORM => SPIRV.ImageFormatRg8
  &Vk.FORMAT_R8G8_SNORM => SPIRV.ImageFormatRg8Snorm
  &Vk.FORMAT_R8G8_UINT => SPIRV.ImageFormatRg8ui
  &Vk.FORMAT_R8G8_SINT => SPIRV.ImageFormatRg8i
  &Vk.FORMAT_R8G8B8A8_UNORM => SPIRV.ImageFormatRgba8
  &Vk.FORMAT_R8G8B8A8_SNORM => SPIRV.ImageFormatRgba8Snorm
  &Vk.FORMAT_R8G8B8A8_UINT => SPIRV.ImageFormatRgba8ui
  &Vk.FORMAT_R8G8B8A8_SINT => SPIRV.ImageFormatRgba8i
  &Vk.FORMAT_A2B10G10R10_UNORM_PACK32 => SPIRV.ImageFormatRgb10A2
  &Vk.FORMAT_A2B10G10R10_UINT_PACK32 => SPIRV.ImageFormatRgb10a2ui
  &Vk.FORMAT_R16_UNORM => SPIRV.ImageFormatR16
  &Vk.FORMAT_R16_SNORM => SPIRV.ImageFormatR16Snorm
  &Vk.FORMAT_R16_UINT => SPIRV.ImageFormatR16ui
  &Vk.FORMAT_R16_SINT => SPIRV.ImageFormatR16i
  &Vk.FORMAT_R16_SFLOAT => SPIRV.ImageFormatR16f
  &Vk.FORMAT_R16G16_UNORM => SPIRV.ImageFormatRg16
  &Vk.FORMAT_R16G16_SNORM => SPIRV.ImageFormatRg16Snorm
  &Vk.FORMAT_R16G16_UINT => SPIRV.ImageFormatRg16ui
  &Vk.FORMAT_R16G16_SINT => SPIRV.ImageFormatRg16i
  &Vk.FORMAT_R16G16_SFLOAT => SPIRV.ImageFormatRg16f
  &Vk.FORMAT_R16G16B16A16_UNORM => SPIRV.ImageFormatRgba16
  &Vk.FORMAT_R16G16B16A16_SNORM => SPIRV.ImageFormatRgba16Snorm
  &Vk.FORMAT_R16G16B16A16_UINT => SPIRV.ImageFormatRgba16ui
  &Vk.FORMAT_R16G16B16A16_SINT => SPIRV.ImageFormatRgba16i
  &Vk.FORMAT_R16G16B16A16_SFLOAT => SPIRV.ImageFormatRgba16f
  &Vk.FORMAT_R32_UINT => SPIRV.ImageFormatR32ui
  &Vk.FORMAT_R32_SINT => SPIRV.ImageFormatR32i
  &Vk.FORMAT_R32_SFLOAT => SPIRV.ImageFormatR32f
  &Vk.FORMAT_R32G32_UINT => SPIRV.ImageFormatRg32ui
  &Vk.FORMAT_R32G32_SINT => SPIRV.ImageFormatRg32i
  &Vk.FORMAT_R32G32_SFLOAT => SPIRV.ImageFormatRg32f
  &Vk.FORMAT_R32G32B32A32_UINT => SPIRV.ImageFormatRgba32ui
  &Vk.FORMAT_R32G32B32A32_SINT => SPIRV.ImageFormatRgba32i
  &Vk.FORMAT_R32G32B32A32_SFLOAT => SPIRV.ImageFormatRgba32f
  &Vk.FORMAT_R64_UINT => SPIRV.ImageFormatR64ui
  &Vk.FORMAT_R64_SINT => SPIRV.ImageFormatR64i
  &Vk.FORMAT_B10G11R11_UFLOAT_PACK32 => SPIRV.ImageFormatR11fG11fB10f
  _ => error("Unknown Vulkan image format $format")
end
