//! # lgpu-hal
//!
//! Micro Hardware Abstraction Layer for `lgpu`.
//! Provides zero-overhead, dynamic-loading access to Vulkan 1.2+, Metal 3, and DirectX 12.

#[cfg(feature = "vulkan")]
pub mod vulkan;
pub mod metal;
pub mod d3d12;

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
    pub format: TextureFormat,
    pub usage: TextureUsage,
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
    #[error("Backend specific error: {0}")]
    BackendError(String),
}
