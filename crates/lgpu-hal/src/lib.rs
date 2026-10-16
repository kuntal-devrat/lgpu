//! # lgpu-hal
//!
//! Vulkan 1.3 hardware abstraction layer for `lgpu`.

#![allow(clippy::all)]

pub mod backend;
#[cfg(feature = "vulkan")]
pub mod vulkan;
pub use backend::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Vulkan,
    Metal,
    DirectX12,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    DiscreteGpu,
    IntegratedGpu,
    VirtualGpu,
    Cpu,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceLimits {
    pub max_buffer_size: u64,
    pub max_texture_dimension_1d: u32,
    pub max_texture_dimension_2d: u32,
    pub max_texture_dimension_3d: u32,
    pub max_texture_array_layers: u32,
    pub min_uniform_buffer_offset_alignment: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub max_compute_workgroup_invocations: u32,
    pub max_compute_workgroup_size: [u32; 3],
}

impl Default for DeviceLimits {
    fn default() -> Self {
        Self {
            max_buffer_size: 256 * 1024 * 1024,
            max_texture_dimension_1d: 8192,
            max_texture_dimension_2d: 8192,
            max_texture_dimension_3d: 2048,
            max_texture_array_layers: 2048,
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 64,
            max_compute_workgroup_invocations: 1024,
            max_compute_workgroup_size: [1024, 1024, 64],
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdapterInfo {
    pub name: String,
    pub backend: Backend,
    pub device_type: DeviceType,
    pub vendor_id: u32,
    pub device_id: u32,
    pub supports_dynamic_rendering: bool,
    pub supports_timeline_semaphores: bool,
    pub supports_descriptor_indexing: bool,
    pub max_compute_workgroup_invocations: u32,
    pub max_compute_workgroup_size: [u32; 3],
    pub limits: DeviceLimits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLocation {
    /// Optimal for GPU read/write. Not accessible directly by CPU.
    GpuOnly,
    /// Host-visible and host-coherent. Optimal for CPU-to-GPU data streaming.
    CpuToGpu,
    /// Host-visible and cached/coherent. Optimal for GPU-to-CPU readback.
    GpuToCpu,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct BufferUsage: u32 {
        const TRANSFER_SRC = 1 << 0;
        const TRANSFER_DST = 1 << 1;
        const UNIFORM      = 1 << 2;
        const STORAGE      = 1 << 3;
        const VERTEX       = 1 << 4;
        const INDEX        = 1 << 5;
        const INDIRECT     = 1 << 6;
    }
}

#[derive(Debug, Clone)]
pub struct BufferDesc<'a> {
    pub label: Option<&'a str>,
    pub size: u64,
    pub usage: BufferUsage,
    pub memory_location: MemoryLocation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    Rgba8Unorm,
    Rgba8Srgb,
    Bgra8Unorm,
    Bgra8Srgb,
    R32Float,
    Rgba16Float,
    Rgba32Float,
    Depth32Float,
}

#[cfg(feature = "vulkan")]
impl TextureFormat {
    pub fn to_vk(&self) -> ash::vk::Format {
        match self {
            TextureFormat::Rgba8Unorm => ash::vk::Format::R8G8B8A8_UNORM,
            TextureFormat::Rgba8Srgb => ash::vk::Format::R8G8B8A8_SRGB,
            TextureFormat::Bgra8Unorm => ash::vk::Format::B8G8R8A8_UNORM,
            TextureFormat::Bgra8Srgb => ash::vk::Format::B8G8R8A8_SRGB,
            TextureFormat::R32Float => ash::vk::Format::R32_SFLOAT,
            TextureFormat::Rgba16Float => ash::vk::Format::R16G16B16A16_SFLOAT,
            TextureFormat::Rgba32Float => ash::vk::Format::R32G32B32A32_SFLOAT,
            TextureFormat::Depth32Float => ash::vk::Format::D32_SFLOAT,
        }
    }

    pub fn from_vk(format: ash::vk::Format) -> Option<Self> {
        match format {
            ash::vk::Format::R8G8B8A8_UNORM => Some(TextureFormat::Rgba8Unorm),
            ash::vk::Format::R8G8B8A8_SRGB => Some(TextureFormat::Rgba8Srgb),
            ash::vk::Format::B8G8R8A8_UNORM => Some(TextureFormat::Bgra8Unorm),
            ash::vk::Format::B8G8R8A8_SRGB => Some(TextureFormat::Bgra8Srgb),
            ash::vk::Format::R32_SFLOAT => Some(TextureFormat::R32Float),
            ash::vk::Format::R16G16B16A16_SFLOAT => Some(TextureFormat::Rgba16Float),
            ash::vk::Format::R32G32B32A32_SFLOAT => Some(TextureFormat::Rgba32Float),
            ash::vk::Format::D32_SFLOAT => Some(TextureFormat::Depth32Float),
            _ => None,
        }
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct TextureUsage: u32 {
        const COPY_SRC         = 1 << 0;
        const COPY_DST         = 1 << 1;
        const TEXTURE_BINDING  = 1 << 2;
        const STORAGE_BINDING  = 1 << 3;
        const RENDER_ATTACHMENT = 1 << 4;
    }
}

#[derive(Debug, Clone)]
pub struct TextureDesc<'a> {
    pub label: Option<&'a str>,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_levels: u32,
    pub sample_count: u32,
    pub format: TextureFormat,
    pub usage: TextureUsage,
}

impl<'a> TextureDesc<'a> {
    pub fn new_2d(width: u32, height: u32, format: TextureFormat, usage: TextureUsage) -> Self {
        Self {
            label: None,
            width,
            height,
            depth: 1,
            mip_levels: 1,
            sample_count: 1,
            format,
            usage,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
}

#[derive(Debug, Clone)]
pub struct SamplerDesc<'a> {
    pub label: Option<&'a str>,
    pub min_filter: FilterMode,
    pub mag_filter: FilterMode,
    pub mipmap_filter: FilterMode,
    pub address_mode_u: AddressMode,
    pub address_mode_v: AddressMode,
    pub address_mode_w: AddressMode,
}

impl Default for SamplerDesc<'_> {
    fn default() -> Self {
        Self {
            label: None,
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadOp<T> {
    Clear(T),
    Load,
    DontCare,
}

pub type ColorLoadOp = LoadOp<[f32; 4]>;
pub type DepthLoadOp = LoadOp<f32>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
}

#[cfg(feature = "vulkan")]
impl PrimitiveTopology {
    pub fn to_vk(&self) -> ash::vk::PrimitiveTopology {
        match self {
            PrimitiveTopology::PointList => ash::vk::PrimitiveTopology::POINT_LIST,
            PrimitiveTopology::LineList => ash::vk::PrimitiveTopology::LINE_LIST,
            PrimitiveTopology::LineStrip => ash::vk::PrimitiveTopology::LINE_STRIP,
            PrimitiveTopology::TriangleList => ash::vk::PrimitiveTopology::TRIANGLE_LIST,
            PrimitiveTopology::TriangleStrip => ash::vk::PrimitiveTopology::TRIANGLE_STRIP,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullMode {
    None,
    Front,
    Back,
}

#[cfg(feature = "vulkan")]
impl CullMode {
    pub fn to_vk(&self) -> ash::vk::CullModeFlags {
        match self {
            CullMode::None => ash::vk::CullModeFlags::NONE,
            CullMode::Front => ash::vk::CullModeFlags::FRONT,
            CullMode::Back => ash::vk::CullModeFlags::BACK,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrontFace {
    Ccw,
    Cw,
}

#[cfg(feature = "vulkan")]
impl FrontFace {
    pub fn to_vk(&self) -> ash::vk::FrontFace {
        match self {
            FrontFace::Ccw => ash::vk::FrontFace::COUNTER_CLOCKWISE,
            FrontFace::Cw => ash::vk::FrontFace::CLOCKWISE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimitiveState {
    pub topology: PrimitiveTopology,
    pub cull_mode: CullMode,
    pub front_face: FrontFace,
}

impl Default for PrimitiveState {
    fn default() -> Self {
        Self {
            topology: PrimitiveTopology::TriangleList,
            cull_mode: CullMode::None,
            front_face: FrontFace::Cw,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstColor,
    OneMinusDstColor,
    DstAlpha,
    OneMinusDstAlpha,
}

#[cfg(feature = "vulkan")]
impl BlendFactor {
    pub fn to_vk(&self) -> ash::vk::BlendFactor {
        match self {
            BlendFactor::Zero => ash::vk::BlendFactor::ZERO,
            BlendFactor::One => ash::vk::BlendFactor::ONE,
            BlendFactor::SrcColor => ash::vk::BlendFactor::SRC_COLOR,
            BlendFactor::OneMinusSrcColor => ash::vk::BlendFactor::ONE_MINUS_SRC_COLOR,
            BlendFactor::SrcAlpha => ash::vk::BlendFactor::SRC_ALPHA,
            BlendFactor::OneMinusSrcAlpha => ash::vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            BlendFactor::DstColor => ash::vk::BlendFactor::DST_COLOR,
            BlendFactor::OneMinusDstColor => ash::vk::BlendFactor::ONE_MINUS_DST_COLOR,
            BlendFactor::DstAlpha => ash::vk::BlendFactor::DST_ALPHA,
            BlendFactor::OneMinusDstAlpha => ash::vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[cfg(feature = "vulkan")]
impl BlendOp {
    pub fn to_vk(&self) -> ash::vk::BlendOp {
        match self {
            BlendOp::Add => ash::vk::BlendOp::ADD,
            BlendOp::Subtract => ash::vk::BlendOp::SUBTRACT,
            BlendOp::ReverseSubtract => ash::vk::BlendOp::REVERSE_SUBTRACT,
            BlendOp::Min => ash::vk::BlendOp::MIN,
            BlendOp::Max => ash::vk::BlendOp::MAX,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendComponent {
    pub src_factor: BlendFactor,
    pub dst_factor: BlendFactor,
    pub operation: BlendOp,
}

impl BlendComponent {
    pub const REPLACE: Self = Self {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::Zero,
        operation: BlendOp::Add,
    };

    pub const OVER: Self = Self {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::OneMinusSrcAlpha,
        operation: BlendOp::Add,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendState {
    pub color: BlendComponent,
    pub alpha: BlendComponent,
}

impl BlendState {
    pub const REPLACE: Self = Self {
        color: BlendComponent::REPLACE,
        alpha: BlendComponent::REPLACE,
    };

    pub const ALPHA_BLENDING: Self = Self {
        color: BlendComponent {
            src_factor: BlendFactor::SrcAlpha,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOp::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOp::Add,
        },
    };

    pub const PREMULTIPLIED_ALPHA_BLENDING: Self = Self {
        color: BlendComponent::OVER,
        alpha: BlendComponent::OVER,
    };

    pub const ADDITIVE: Self = Self {
        color: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            operation: BlendOp::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            operation: BlendOp::Add,
        },
    };
}

#[derive(Debug, Clone)]
pub struct GraphicsPipelineDesc<'a> {
    pub label: Option<&'a str>,
    pub vs_entry: &'a str,
    pub fs_entry: &'a str,
    pub color_format: TextureFormat,
    pub depth_format: Option<TextureFormat>,
    pub depth_test: bool,
    pub depth_write: bool,
    pub sample_count: u32,
    pub primitive: PrimitiveState,
    pub blend: Option<BlendState>,
}

impl<'a> GraphicsPipelineDesc<'a> {
    pub fn new(vs_entry: &'a str, fs_entry: &'a str, color_format: TextureFormat) -> Self {
        Self {
            label: None,
            vs_entry,
            fs_entry,
            color_format,
            depth_format: None,
            depth_test: false,
            depth_write: false,
            sample_count: 1,
            primitive: PrimitiveState::default(),
            blend: None,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum HalError {
    #[error("Dynamic loader failed to load driver library: {0}")]
    LibraryLoadError(String),
    #[error("No compatible graphics adapter was found")]
    NoAdapterFound,
    #[error("Device creation failed: {0}")]
    DeviceCreationFailed(String),
    #[error("Out of memory: {0}")]
    OutOfMemory(String),
    #[error("Shader module compilation failed: {0}")]
    ShaderError(String),
    #[error("Pipeline creation failed: {0}")]
    PipelineError(String),
    #[error("Command recording or submission error: {0}")]
    CommandError(String),
    #[error("Map buffer error: {0}")]
    MapError(String),
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("Backend specific error: {0}")]
    BackendError(String),
}
