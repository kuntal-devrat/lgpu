//! Per-frame attachment management for surface rendering.
//!
//! [`FrameRenderer`] owns the multisampled color target and depth buffer that a window's
//! swapchain image needs. It recreates them when the surface is resized so callers only have
//! to describe their depth format and sample count once.

use crate::{
    ColorLoadOp, CommandEncoder, DepthLoadOp, Device, Error, HalError, Queue, RenderPass,
    RenderPassDesc, Surface, SurfaceConfiguration, SurfaceTexture, Texture, TextureDesc,
    TextureDimension, TextureFormat, TextureUsage,
};

/// Attachment requirements for a window's frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameConfig {
    /// Depth format to attach, or `None` for 2D-only rendering.
    pub depth_format: Option<TextureFormat>,
    /// MSAA sample count used for the color target. `1` disables multisampling.
    pub sample_count: u32,
}

impl Default for FrameConfig {
    fn default() -> Self {
        Self {
            depth_format: Some(TextureFormat::Depth32Float),
            sample_count: 1,
        }
    }
}

impl FrameConfig {
    /// Configures a depth attachment without multisampling.
    pub fn with_depth(depth_format: TextureFormat) -> Self {
        Self {
            depth_format: Some(depth_format),
            sample_count: 1,
        }
    }

    /// Configures multisampled rendering with an optional depth attachment.
    pub fn multisampled(sample_count: u32, depth_format: Option<TextureFormat>) -> Self {
        Self {
            depth_format,
            sample_count,
        }
    }
}

/// The attachments for a single frame, sized to the current surface image.
#[derive(Debug)]
pub struct FrameAttachments {
    /// The color target draws write to. This is the multisampled image when MSAA is enabled,
    /// and otherwise a borrowed view of the acquired surface image.
    pub color: Texture,
    /// Single-sampled resolve target. Only present when MSAA is enabled.
    pub resolve: Option<Texture>,
    /// Depth/stencil attachment, when one was requested.
    pub depth: Option<Texture>,
    pub width: u32,
    pub height: u32,
    pub sample_count: u32,
}

impl FrameAttachments {
    /// Begins a scoped render pass that writes to these attachments and resolves them when
    /// multisampling is active.
    pub fn begin<'a>(
        &'a self,
        encoder: &'a mut CommandEncoder,
        color_load_op: ColorLoadOp,
        depth_clear: f32,
    ) -> RenderPass<'a> {
        let desc = RenderPassDesc {
            color_texture: &self.color,
            color_load_op,
            depth_attachment: self
                .depth
                .as_ref()
                .map(|depth| (depth, DepthLoadOp::Clear(depth_clear))),
        };
        encoder.begin_render_pass_with_resolve(desc, self.resolve.as_ref())
    }
}

/// Recreates the depth and MSAA attachments whenever the surface extent changes.
#[derive(Debug)]
pub struct FrameRenderer {
    config: FrameConfig,
    width: u32,
    height: u32,
    color: Option<Texture>,
    depth: Option<Texture>,
}

impl FrameRenderer {
    pub fn new(config: FrameConfig) -> Self {
        Self {
            config,
            width: 0,
            height: 0,
            color: None,
            depth: None,
        }
    }

    pub fn config(&self) -> FrameConfig {
        self.config
    }

    /// Builds the attachments for `target`, recreating them if its extent changed.
    pub fn prepare(
        &mut self,
        device: &Device,
        target: &Texture,
    ) -> Result<FrameAttachments, Error> {
        let sample_count = self.config.sample_count.max(1);
        if target.width == 0 || target.height == 0 {
            return Err(Error::Hal(lgpu_hal::HalError::NotReady(
                "FrameRenderer target has a zero extent".to_string(),
            )));
        }

        let resized = self.width != target.width || self.height != target.height;
        if resized || self.color.is_some() != (sample_count > 1) {
            self.width = target.width;
            self.height = target.height;

            self.color = if sample_count > 1 {
                Some(device.create_texture(&TextureDesc {
                    label: Some("lgpu_frame_msaa_color"),
                    width: target.width,
                    height: target.height,
                    depth: 1,
                    array_layers: 1,
                    dimension: TextureDimension::D2,
                    mip_levels: 1,
                    sample_count,
                    format: target.format,
                    usage: TextureUsage::RENDER_ATTACHMENT,
                })?)
            } else {
                None
            };

            self.depth = match self.config.depth_format {
                Some(format) => Some(device.create_texture(&TextureDesc {
                    label: Some("lgpu_frame_depth"),
                    width: target.width,
                    height: target.height,
                    depth: 1,
                    array_layers: 1,
                    dimension: TextureDimension::D2,
                    mip_levels: 1,
                    sample_count,
                    format,
                    usage: TextureUsage::RENDER_ATTACHMENT,
                })?),
                None => None,
            };
        }

        Ok(FrameAttachments {
            color: match &self.color {
                Some(color) => color.clone_unowned(),
                None => target.clone_unowned(),
            },
            resolve: self.color.as_ref().map(|_| target.clone_unowned()),
            depth: self.depth.as_ref().map(|depth| depth.clone_unowned()),
            width: target.width,
            height: target.height,
            sample_count,
        })
    }

    /// Releases the cached attachments; the next [`FrameRenderer::prepare`] recreates them.
    pub fn clear(&mut self) {
        self.color = None;
        self.depth = None;
        self.width = 0;
        self.height = 0;
    }
}

/// Drives a window surface end to end: acquires an image, reconfigures the swapchain when it is
/// out of date, rebuilds the attachments when the size changes, submits the recorded frame and
/// presents it.
#[derive(Debug)]
pub struct SurfaceFrames {
    config: SurfaceConfiguration,
    renderer: FrameRenderer,
}

/// An acquired surface image together with its per-frame attachments.
pub struct SurfaceFrame<'a> {
    surface: &'a mut Surface,
    config: &'a SurfaceConfiguration,
    attachments: FrameAttachments,
    surface_texture: SurfaceTexture,
}

impl SurfaceFrames {
    pub fn new(config: SurfaceConfiguration, frame: FrameConfig) -> Self {
        Self {
            config,
            renderer: FrameRenderer::new(frame),
        }
    }

    pub fn config(&self) -> &SurfaceConfiguration {
        &self.config
    }

    pub fn renderer(&self) -> &FrameRenderer {
        &self.renderer
    }

    /// Replaces the stored surface configuration, for example after a resize.
    pub fn set_config(&mut self, config: SurfaceConfiguration) {
        self.config = config;
    }

    /// Mutable access to the stored configuration; call `Surface::configure` afterwards.
    pub fn config_mut(&mut self) -> &mut SurfaceConfiguration {
        &mut self.config
    }

    /// Acquires the next image, reconfiguring the surface once if the swapchain is stale.
    pub fn acquire<'a>(
        &'a mut self,
        device: &Device,
        surface: &'a mut Surface,
    ) -> Result<SurfaceFrame<'a>, Error> {
        let surface_texture = match surface.get_current_texture() {
            Ok(texture) => texture,
            Err(Error::Hal(HalError::OutOfDate(_))) | Err(Error::Hal(HalError::Suboptimal(_))) => {
                surface.configure(device, &self.config)?;
                surface.get_current_texture()?
            }
            Err(e) => return Err(e),
        };

        let attachments = self.renderer.prepare(device, &surface_texture.texture)?;
        Ok(SurfaceFrame {
            surface,
            config: &self.config,
            attachments,
            surface_texture,
        })
    }
}

impl SurfaceFrame<'_> {
    pub fn attachments(&self) -> &FrameAttachments {
        &self.attachments
    }

    pub fn size(&self) -> (u32, u32) {
        (self.attachments.width, self.attachments.height)
    }

    /// Begins a render pass that targets the frame's color (and depth) attachments and resolves
    /// into the swapchain image when MSAA is active.
    pub fn begin<'a>(
        &'a self,
        encoder: &'a mut CommandEncoder,
        color_load_op: ColorLoadOp,
        depth_clear: f32,
    ) -> RenderPass<'a> {
        self.attachments.begin(encoder, color_load_op, depth_clear)
    }

    /// Recreates the swapchain from the stored configuration before this frame is submitted.
    pub fn reconfigure(&mut self, device: &Device) -> Result<(), Error> {
        self.surface.configure(device, self.config)
    }

    /// Submits the recorded frame, waits for the queue, and presents the acquired image.
    pub fn present(self, queue: &Queue, encoder: CommandEncoder) -> Result<(), Error> {
        queue.submit_surface_work(encoder.finish(), &self.surface_texture)?;
        self.surface_texture.present(queue)
    }
}
