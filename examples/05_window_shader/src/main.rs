//! # lgpu Interactive Window Demo
//!
//! Renders live animated 3D polygons, procedural raymarched shaders, and dynamic
//! geometric visuals using native Vulkan 1.3 Swapchain and the Lean GPU (lgpu) abstraction.

mod shaders;

use std::sync::Arc;
use std::time::Instant;

use lgpu::{
    Device, GraphicsPipeline, Instance, PresentMode, Queue, Surface, SurfaceConfiguration,
    TextureFormat, TextureUsage,
};
use winit::{
    dpi::LogicalSize,
    event::{ElementState, Event, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::WindowBuilder,
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    time: f32,
    aspect: f32,
    rot_x: f32,
    rot_y: f32,
    rot_z: f32,
    mode: u32,
    param1: f32,
    param2: f32,
}

struct RenderState {
    device: Device,
    queue: Queue,
    surface: Surface,
    polygon_pipeline: GraphicsPipeline,
    procedural_pipeline: GraphicsPipeline,
    width: u32,
    height: u32,
}

impl RenderState {
    fn new(
        device: Device,
        queue: Queue,
        mut surface: Surface,
        width: u32,
        height: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let config = SurfaceConfiguration {
            usage: TextureUsage::RENDER_ATTACHMENT,
            format: TextureFormat::Bgra8Unorm,
            width,
            height,
            present_mode: PresentMode::Mailbox,
        };
        surface.configure(&device, &config)?;

        let polygon_shader = device.create_shader_spirv(shaders::POLYGON_3D_SPV)?;
        let polygon_pipeline = device.create_graphics_pipeline(
            &polygon_shader,
            "vs_main",
            "fs_main",
            TextureFormat::Bgra8Unorm,
        )?;

        let procedural_shader = device.create_shader_spirv(shaders::PROCEDURAL_SHADER_SPV)?;
        let procedural_pipeline = device.create_graphics_pipeline(
            &procedural_shader,
            "vs_main",
            "fs_main",
            TextureFormat::Bgra8Unorm,
        )?;

        Ok(Self {
            device,
            queue,
            surface,
            polygon_pipeline,
            procedural_pipeline,
            width,
            height,
        })
    }

    fn resize(&mut self, new_width: u32, new_height: u32) -> Result<(), Box<dyn std::error::Error>> {
        if new_width == 0 || new_height == 0 || (new_width == self.width && new_height == self.height) {
            return Ok(());
        }
        self.width = new_width;
        self.height = new_height;
        let config = SurfaceConfiguration {
            usage: TextureUsage::RENDER_ATTACHMENT,
            format: TextureFormat::Bgra8Unorm,
            width: new_width,
            height: new_height,
            present_mode: PresentMode::Mailbox,
        };
        self.surface.configure(&self.device, &config)?;
        Ok(())
    }

    fn render(
        &mut self,
        pc: &PushConstants,
        mode: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => {
                let config = SurfaceConfiguration {
                    usage: TextureUsage::RENDER_ATTACHMENT,
                    format: TextureFormat::Bgra8Unorm,
                    width: self.width,
                    height: self.height,
                    present_mode: PresentMode::Mailbox,
                };
                self.surface.configure(&self.device, &config)?;
                self.surface.get_current_texture()?
            }
        };

        let mut encoder = self.device.create_command_encoder()?;

        if mode == 0 {
            // Mode 0: 3D Animated Shaded Polygons (Cube/Prism, 36 vertices)
            encoder.draw(
                &frame.texture,
                &self.polygon_pipeline,
                36,
                bytemuck::bytes_of(pc),
                [0.06, 0.06, 0.10, 1.0], // Dark aesthetic deep navy background
            )?;
        } else {
            // Mode 1: 3D Raymarched Neon Torus
            // Mode 2: Cosmic Psychedelic Plasma Waveform
            encoder.draw(
                &frame.texture,
                &self.procedural_pipeline,
                3,
                bytemuck::bytes_of(pc),
                [0.0, 0.0, 0.0, 1.0],
            )?;
        }

        encoder.transition_to_present(&frame.texture);
        self.queue.submit_surface(encoder.finish(), frame)?;
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("      lgpu: Modern Cross-Platform Ultra-Fast GPU Engine      ");
    println!("   Live Window Demo: 3D Polygons & Real-Time Shaders       ");
    println!("============================================================");

    // 1. Initialize Vulkan 1.3 Device via lgpu
    let instance = Instance::new()?;
    let adapters = instance.enumerate_adapters()?;
    if adapters.is_empty() {
        return Err("No compatible GPU adapter found".into());
    }
    let adapter = &adapters[0];
    let gpu_info = adapter.info();
    println!("GPU Device: {} ({:?})", gpu_info.name, gpu_info.device_type);
    println!("Backend: {:?}, Dynamic Rendering: {}", gpu_info.backend, gpu_info.supports_dynamic_rendering);

    let (device, queue) = adapter.request_device(&instance)?;

    // 2. Setup Winit Event Loop and Window
    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("lgpu - 3D Polygons & Shaders [Press 1, 2, 3 or Space]")
            .with_inner_size(LogicalSize::new(800.0, 600.0))
            .with_min_inner_size(LogicalSize::new(320.0, 240.0))
            .build(&event_loop)?,
    );

    let surface = instance.create_surface(&window)?;
    let initial_size = window.inner_size();
    let mut render_state = RenderState::new(
        device,
        queue,
        surface,
        initial_size.width.max(1),
        initial_size.height.max(1),
    )?;

    println!("\nInteractive Controls:");
    println!("  [1]     : 3D Rotating Polygons (Shaded Polyhedron with Dynamic Lighting)");
    println!("  [2]     : Cyberpunk Raymarched 3D Torus (Volumetric SDF & Neon Reflections)");
    println!("  [3]     : Cosmic Psychedelic Plasma Waveform (Mathematical Fluid Dynamics)");
    println!("  [Space] : Cycle to next Shader Mode");
    println!("  [Drag]  : Click & Drag mouse to manually rotate 3D Geometry");
    println!("  [Esc]   : Exit Demo");
    println!("------------------------------------------------------------\n");

    let start_time = Instant::now();
    let mut last_frame = Instant::now();
    let mut frame_count = 0u32;
    let mut last_fps_time = Instant::now();

    let mut current_mode = 0u32; // 0 = 3D Polygons, 1 = Raymarched Torus, 2 = Cosmic Plasma
    let mut rot_x = 0.5f32;
    let mut rot_y = 0.8f32;
    let mut rot_z = 0.2f32;
    let mut mouse_down = false;
    let mut last_mouse_pos = (0.0f32, 0.0f32);

    let mode_names = [
        "Mode [1]: 3D Rotating Polygons",
        "Mode [2]: Cyberpunk Raymarched 3D Torus",
        "Mode [3]: Cosmic Psychedelic Plasma",
    ];

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        if let Event::WindowEvent { event, .. } = event {
            match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(physical_size) => {
                    let w = physical_size.width.max(1);
                    let h = physical_size.height.max(1);
                    if let Err(e) = render_state.resize(w, h) {
                        eprintln!("Resize error: {e}");
                    }
                    window.request_redraw();
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            state: ElementState::Pressed,
                            logical_key,
                            ..
                        },
                    ..
                } => match logical_key {
                    Key::Named(NamedKey::Escape) => elwt.exit(),
                    Key::Named(NamedKey::Space) => {
                        current_mode = (current_mode + 1) % 3;
                        println!("Switched to {}", mode_names[current_mode as usize]);
                    }
                    Key::Character(ref ch) => match ch.as_str() {
                        "1" => {
                            current_mode = 0;
                            println!("Switched to {}", mode_names[0]);
                        }
                        "2" => {
                            current_mode = 1;
                            println!("Switched to {}", mode_names[1]);
                        }
                        "3" => {
                            current_mode = 2;
                            println!("Switched to {}", mode_names[2]);
                        }
                        _ => {}
                    },
                    _ => {}
                },
                WindowEvent::MouseInput {
                    state,
                    button: winit::event::MouseButton::Left,
                    ..
                } => {
                    mouse_down = state == ElementState::Pressed;
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let pos = (position.x as f32, position.y as f32);
                    if mouse_down {
                        let dx = pos.0 - last_mouse_pos.0;
                        let dy = pos.1 - last_mouse_pos.1;
                        rot_y += dx * 0.01;
                        rot_x += dy * 0.01;
                    }
                    last_mouse_pos = pos;
                }
                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let dt = now.duration_since(last_frame).as_secs_f32();
                    last_frame = now;

                    let total_time = start_time.elapsed().as_secs_f32();

                    // Auto-rotate when not manually dragging
                    if !mouse_down {
                        rot_x += 0.8 * dt;
                        rot_y += 1.2 * dt;
                        rot_z += 0.5 * dt;
                    }

                    // FPS calculation
                    frame_count += 1;
                    if last_fps_time.elapsed().as_secs_f32() >= 1.0 {
                        let fps = frame_count;
                        frame_count = 0;
                        last_fps_time = Instant::now();

                        let title = format!(
                            "lgpu | {} | {} FPS | GPU: {}",
                            mode_names[current_mode as usize],
                            fps,
                            gpu_info.name
                        );
                        window.set_title(&title);
                        println!("[lgpu Window] Rendered {} FPS | {}", fps, mode_names[current_mode as usize]);
                    }

                    // Render GPU Frame
                    let w = render_state.width;
                    let h = render_state.height;
                    let aspect = w as f32 / h as f32;

                    let pc = PushConstants {
                        time: total_time,
                        aspect,
                        rot_x,
                        rot_y,
                        rot_z,
                        mode: current_mode,
                        param1: 0.0,
                        param2: 0.0,
                    };

                    if let Err(e) = render_state.render(&pc, current_mode) {
                        eprintln!("Render error: {e}");
                    }

                    window.request_redraw();
                }
                _ => {}
            }
        }
    })?;

    Ok(())
}
