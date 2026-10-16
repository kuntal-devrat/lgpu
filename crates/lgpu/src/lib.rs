//! # lgpu (Lean GPU)
//!
//! An ultra-lightweight, bindless-first cross-platform GPU abstraction for Rust and C.
//! Designed for sub-second compile times, minimal binary footprint, and maximum performance.

#![allow(clippy::too_many_arguments)]

#[cfg(not(feature = "vulkan"))]
compile_error!(
    "lgpu requires the 'vulkan' feature to be enabled as it is the native production backend."
);

#[cfg(feature = "vulkan")]
pub mod buffer;
#[cfg(feature = "vulkan")]
pub mod command;
#[cfg(feature = "vulkan")]
pub mod device;
#[cfg(feature = "vulkan")]
pub mod instance;
#[cfg(feature = "vulkan")]
pub mod pipeline;
#[cfg(feature = "vulkan")]
pub mod surface;
#[cfg(feature = "vulkan")]
pub mod sync;
#[cfg(feature = "vulkan")]
pub mod texture;

#[cfg(feature = "vulkan")]
pub use buffer::*;
#[cfg(feature = "vulkan")]
pub use command::*;
#[cfg(feature = "vulkan")]
pub use device::*;
#[cfg(feature = "vulkan")]
pub use instance::*;
#[cfg(feature = "vulkan")]
pub use pipeline::*;
#[cfg(feature = "vulkan")]
pub use surface::*;
#[cfg(feature = "vulkan")]
pub use sync::*;
#[cfg(feature = "vulkan")]
pub use texture::*;

pub use lgpu_hal::{
    AdapterInfo, AddressMode, Backend, BlendComponent, BlendFactor, BlendOp, BlendState,
    BufferDesc, BufferUsage, ColorLoadOp, CullMode, DepthLoadOp, DeviceLimits, DeviceType,
    FilterMode, FrontFace, GraphicsPipelineDesc, HalError, LoadOp, MemoryLocation, PrimitiveState,
    PrimitiveTopology, SamplerDesc, TextureDesc, TextureFormat, TextureUsage,
};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Hardware Abstraction Layer error: {0}")]
    Hal(#[from] HalError),
    #[error("Buffer error: {0}")]
    Buffer(String),
}
