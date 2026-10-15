//! # lgpu (Lean GPU)
//!
//! An ultra-lightweight, bindless-first cross-platform GPU abstraction for Rust and C.
//! Designed for sub-second compile times, minimal binary footprint, and maximum performance.

pub mod buffer;
pub mod command;
pub mod device;
pub mod instance;
pub mod pipeline;
pub mod surface;
pub mod sync;
pub mod texture;

pub use buffer::*;
pub use command::*;
pub use device::*;
pub use instance::*;
pub use pipeline::*;
pub use surface::*;
pub use sync::*;
pub use texture::*;

pub use lgpu_hal::{
    AdapterInfo, AddressMode, Backend, BlendComponent, BlendFactor, BlendOp, BlendState,
    BufferDesc, BufferUsage, ColorLoadOp, CullMode, DepthLoadOp, DeviceType, FilterMode,
    FrontFace, GraphicsPipelineDesc, HalError, LoadOp, MemoryLocation, PrimitiveState,
    PrimitiveTopology, SamplerDesc, TextureDesc, TextureFormat, TextureUsage,
};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Hardware Abstraction Layer error: {0}")]
    Hal(#[from] HalError),
    #[error("Buffer error: {0}")]
    Buffer(String),
}
