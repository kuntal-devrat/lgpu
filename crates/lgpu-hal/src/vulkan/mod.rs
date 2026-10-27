//! Vulkan backend implementation for lgpu-hal
//!
//! Uses zero-cost dynamic loading via `ash` to interface with `libvulkan.so.1`,
//! `vulkan-1.dll`, or `MoltenVK`.

pub use ash::vk;
pub use ash::vk::Pipeline as RawPipeline;

pub mod allocator;
pub mod bindless;
pub mod buffer;
pub mod command;
pub mod device;
pub mod instance;
pub mod pipeline;
pub mod staging;
pub mod swapchain;
pub mod sync;
pub mod texture;

pub use buffer::*;
pub use command::*;
pub use device::*;
pub use instance::*;
pub use pipeline::*;
pub use staging::*;
pub use swapchain::*;
pub use sync::*;
pub use texture::*;
