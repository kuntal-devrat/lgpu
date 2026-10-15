//! Vulkan shader module and pipeline creation.

use ash::vk;
use std::ffi::CString;

use super::VulkanDevice;
use crate::{HalError, TextureFormat};

pub use ash::vk::Pipeline as RawPipeline;

pub struct VulkanShader {
    pub(crate) module: vk::ShaderModule,
}

unsafe impl Send for VulkanShader {}
unsafe impl Sync for VulkanShader {}

pub struct VulkanComputePipeline {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) descriptor_set_layout: vk::DescriptorSetLayout,
    pub(crate) descriptor_pool: vk::DescriptorPool,
    pub(crate) descriptor_set: vk::DescriptorSet,
}

unsafe impl Send for VulkanComputePipeline {}
unsafe impl Sync for VulkanComputePipeline {}

impl VulkanDevice {
    pub fn create_shader_from_spirv(&self, spirv_words: &[u32]) -> Result<VulkanShader, HalError> {
        const SPIRV_MAGIC: u32 = 0x07230203;
        if spirv_words.len() < 5 || spirv_words[0] != SPIRV_MAGIC {
            return Err(HalError::ShaderError(
                "Invalid SPIR-V bytecode: missing or invalid magic header".to_string(),
            ));
        }

        let create_info = vk::ShaderModuleCreateInfo::default().code(spirv_words);
        let module = unsafe {
            self.device
                .create_shader_module(&create_info, None)
                .map_err(|e| HalError::ShaderError(e.to_string()))?
        };
        Ok(VulkanShader { module })
    }

    pub fn create_compute_pipeline(
        &self,
        shader: &VulkanShader,
        buffer_count: u32,
    ) -> Result<VulkanComputePipeline, HalError> {
        let mut bindings = Vec::with_capacity(buffer_count as usize);
        for i in 0..buffer_count {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
        }

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&layout_info, None)
                .map_err(|e| HalError::PipelineError(e.to_string()))?
        };

        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);

        let layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| HalError::PipelineError(e.to_string()))?
        };

        let entry_point = CString::new("main").unwrap();
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader.module)
            .name(&entry_point);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(layout);

        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(self.pipeline_cache, &[pipeline_info], None)
                .map_err(|(_, e)| HalError::PipelineError(e.to_string()))?
        };

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(buffer_count.max(1))];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);

        let descriptor_pool = unsafe {
            self.device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| HalError::PipelineError(e.to_string()))?
        };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);

        let descriptor_sets = unsafe {
            self.device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| HalError::PipelineError(e.to_string()))?
        };

        Ok(VulkanComputePipeline {
            pipeline: pipelines[0],
            layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set: descriptor_sets[0],
        })
    }

    pub fn create_bindless_compute_pipeline(
        &self,
        shader: &VulkanShader,
    ) -> Result<vk::Pipeline, HalError> {
        let heap = self.bindless_heap.as_ref().ok_or_else(|| {
            HalError::BackendError("Bindless descriptor heap is not enabled".to_string())
        })?;

        let entry_point = CString::new("main").unwrap();
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader.module)
            .name(&entry_point);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(heap.pipeline_layout);

        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(self.pipeline_cache, &[pipeline_info], None)
                .map_err(|(_, e)| HalError::PipelineError(e.to_string()))?
        };

        Ok(pipelines[0])
    }

    pub fn destroy_raw_pipeline(&self, pipeline: vk::Pipeline) {
        unsafe {
            self.device.destroy_pipeline(pipeline, None);
        }
    }

    pub fn create_graphics_pipeline(
        &self,
        shader: &VulkanShader,
        vs_entry: &str,
        fs_entry: &str,
        color_format: TextureFormat,
    ) -> Result<vk::Pipeline, HalError> {
        self.create_graphics_pipeline_with_depth(
            shader,
            vs_entry,
            fs_entry,
            color_format,
            None,
            false,
            false,
        )
    }

    pub fn create_graphics_pipeline_with_depth(
        &self,
        shader: &VulkanShader,
        vs_entry: &str,
        fs_entry: &str,
        color_format: TextureFormat,
        depth_format: Option<TextureFormat>,
        depth_test: bool,
        depth_write: bool,
    ) -> Result<vk::Pipeline, HalError> {
        self.create_graphics_pipeline_with_depth_and_samples(
            shader,
            vs_entry,
            fs_entry,
            color_format,
            depth_format,
            depth_test,
            depth_write,
            1,
        )
    }

    pub fn create_graphics_pipeline_with_depth_and_samples(
        &self,
        shader: &VulkanShader,
        vs_entry: &str,
        fs_entry: &str,
        color_format: TextureFormat,
        depth_format: Option<TextureFormat>,
        depth_test: bool,
        depth_write: bool,
        sample_count: u32,
    ) -> Result<vk::Pipeline, HalError> {
        let desc = crate::GraphicsPipelineDesc {
            label: None,
            vs_entry,
            fs_entry,
            color_format,
            depth_format,
            depth_test,
            depth_write,
            sample_count,
            primitive: crate::PrimitiveState::default(),
            blend: None,
        };
        self.create_graphics_pipeline_with_desc(shader, &desc)
    }

    pub fn create_graphics_pipeline_with_desc(
        &self,
        shader: &VulkanShader,
        desc: &crate::GraphicsPipelineDesc,
    ) -> Result<vk::Pipeline, HalError> {
        let heap = self.bindless_heap.as_ref().ok_or_else(|| {
            HalError::BackendError("Bindless descriptor heap is not enabled".to_string())
        })?;

        let vk_format = desc.color_format.to_vk();

        let vs_name = CString::new(desc.vs_entry).unwrap();
        let fs_name = CString::new(desc.fs_entry).unwrap();

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(shader.module)
                .name(&vs_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(shader.module)
                .name(&fs_name),
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(desc.primitive.topology.to_vk())
            .primitive_restart_enable(false);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(desc.primitive.cull_mode.to_vk())
            .front_face(desc.primitive.front_face.to_vk())
            .depth_bias_enable(false);

        let vk_samples = match desc.sample_count {
            2 => vk::SampleCountFlags::TYPE_2,
            4 => vk::SampleCountFlags::TYPE_4,
            8 => vk::SampleCountFlags::TYPE_8,
            16 => vk::SampleCountFlags::TYPE_16,
            32 => vk::SampleCountFlags::TYPE_32,
            64 => vk::SampleCountFlags::TYPE_64,
            _ => vk::SampleCountFlags::TYPE_1,
        };

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk_samples);

        let color_blend_attachment = if let Some(b) = desc.blend {
            [vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(true)
                .src_color_blend_factor(b.color.src_factor.to_vk())
                .dst_color_blend_factor(b.color.dst_factor.to_vk())
                .color_blend_op(b.color.operation.to_vk())
                .src_alpha_blend_factor(b.alpha.src_factor.to_vk())
                .dst_alpha_blend_factor(b.alpha.dst_factor.to_vk())
                .alpha_blend_op(b.alpha.operation.to_vk())]
        } else {
            [vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false)]
        };

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&color_blend_attachment);

        let color_formats = [vk_format];
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

        let vk_depth_format = match desc.depth_format {
            Some(TextureFormat::Depth32Float) => Some(vk::Format::D32_SFLOAT),
            Some(f) => {
                return Err(HalError::BackendError(format!(
                    "Unsupported depth format: {f:?}"
                )))
            }
            None => None,
        };

        if let Some(df) = vk_depth_format {
            rendering_info = rendering_info.depth_attachment_format(df);
        }

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(desc.depth_test)
            .depth_write_enable(desc.depth_write)
            .depth_compare_op(if desc.depth_test {
                vk::CompareOp::LESS_OR_EQUAL
            } else {
                vk::CompareOp::ALWAYS
            })
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let mut pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state_info)
            .layout(heap.pipeline_layout)
            .push_next(&mut rendering_info);

        if desc.depth_format.is_some() || desc.depth_test {
            pipeline_info = pipeline_info.depth_stencil_state(&depth_stencil);
        }

        let pipelines = unsafe {
            self.device
                .create_graphics_pipelines(self.pipeline_cache, &[pipeline_info], None)
                .map_err(|(_, e)| {
                    HalError::PipelineError(format!("create_graphics_pipelines failed: {e}"))
                })?
        };

        Ok(pipelines[0])
    }

    pub fn destroy_shader(&self, shader: VulkanShader) {
        unsafe {
            self.device.destroy_shader_module(shader.module, None);
        }
    }

    pub fn destroy_compute_pipeline(&self, pipeline: VulkanComputePipeline) {
        unsafe {
            self.device
                .destroy_descriptor_pool(pipeline.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(pipeline.descriptor_set_layout, None);
            self.device.destroy_pipeline(pipeline.pipeline, None);
            self.device.destroy_pipeline_layout(pipeline.layout, None);
        }
    }
}
