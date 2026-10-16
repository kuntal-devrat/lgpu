//! GPU shaders, compute pipelines, and graphics pipelines.

pub use lgpu_hal::{
    BlendComponent, BlendFactor, BlendOp, BlendState, CullMode, FrontFace, GraphicsPipelineDesc,
    PrimitiveState, PrimitiveTopology,
};
use std::sync::Arc;

pub struct Shader {
    pub(crate) hal_shader: Option<lgpu_hal::vulkan::VulkanShader>,
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
}

impl std::fmt::Debug for Shader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shader").finish()
    }
}

pub struct ComputePipeline {
    pub(crate) hal_pipeline: Option<lgpu_hal::vulkan::VulkanComputePipeline>,
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
}

impl std::fmt::Debug for ComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputePipeline").finish()
    }
}

pub struct BindlessComputePipeline {
    pub(crate) pipeline: Option<lgpu_hal::vulkan::RawPipeline>,
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
}

impl std::fmt::Debug for BindlessComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BindlessComputePipeline").finish()
    }
}

pub struct GraphicsPipeline {
    pub(crate) pipeline: Option<lgpu_hal::vulkan::RawPipeline>,
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
}

impl std::fmt::Debug for GraphicsPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphicsPipeline").finish()
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        if let Some(shader) = self.hal_shader.take() {
            self.vulkan_device.destroy_shader(shader);
        }
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        if let Some(pipeline) = self.hal_pipeline.take() {
            self.vulkan_device.destroy_compute_pipeline(pipeline);
        }
    }
}

impl Drop for BindlessComputePipeline {
    fn drop(&mut self) {
        if let Some(pipeline) = self.pipeline.take() {
            self.vulkan_device.destroy_raw_pipeline(pipeline);
        }
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        if let Some(pipeline) = self.pipeline.take() {
            self.vulkan_device.destroy_raw_pipeline(pipeline);
        }
    }
}
