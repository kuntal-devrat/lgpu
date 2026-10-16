//! # lgpu Real-Time Voxel Renderer
//!
//! A real-time voxel raymarching engine powered by Amanatides & Woo Fast Voxel DDA,
//! dynamic rendering, and Push Constants using the `lgpu` modern Vulkan 1.3 layer.

mod compiler;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use lgpu::{
    Device, GraphicsPipeline, Instance, PresentMode, Queue, Surface, SurfaceConfiguration,
    TextureFormat, TextureUsage,
};
use winit::{
    dpi::LogicalSize,
    event::{ElementState, Event, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::WindowBuilder,
};

const VOXEL_SHADER_SRC: &str = include_str!("../shaders/voxel.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VoxelPushConstants {
    cam_pos: [f32; 3],
    time: f32,
    cam_dir: [f32; 3],
    aspect: f32,
    cam_up: [f32; 3],
    mode: u32,
    render_mode: u32,
    sun_angle: f32,
    fov: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct Camera {
    pos: [f32; 3],
    yaw: f32,
    pitch: f32,
    fov: f32,
}

impl Camera {
    fn new(pos: [f32; 3], yaw: f32, pitch: f32) -> Self {
        Self {
            pos,
            yaw,
            pitch,
            fov: 70.0,
        }
    }

    fn forward_vector(&self) -> [f32; 3] {
        let yaw_rad = self.yaw.to_radians();
        let pitch_rad = self.pitch.to_radians();
        let cp = pitch_rad.cos();
        [yaw_rad.cos() * cp, pitch_rad.sin(), yaw_rad.sin() * cp]
    }

    fn right_vector(&self) -> [f32; 3] {
        let yaw_rad = self.yaw.to_radians();
        [-yaw_rad.sin(), 0.0, yaw_rad.cos()]
    }

    fn up_vector(&self) -> [f32; 3] {
        [0.0, 1.0, 0.0]
    }
}

struct VoxelRenderer {
    device: Device,
    queue: Queue,
    surface: Surface,
    pipeline: GraphicsPipeline,
    width: u32,
    height: u32,
}

impl VoxelRenderer {
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

        // Compile WGSL to SPIR-V in ~2 milliseconds using Naga
        let spv_words = compiler::compile_wgsl_to_spirv(VOXEL_SHADER_SRC)
            .map_err(|e| format!("Failed to compile voxel WGSL: {e}"))?;

        let shader = device.create_shader_spirv(&spv_words)?;
        let pipeline = device.create_graphics_pipeline(
            &shader,
            "vs_main",
            "fs_main",
            TextureFormat::Bgra8Unorm,
        )?;

        Ok(Self {
            device,
            queue,
            surface,
            pipeline,
            width,
            height,
        })
    }

    fn resize(
        &mut self,
        new_width: u32,
        new_height: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if new_width == 0
            || new_height == 0
            || (new_width == self.width && new_height == self.height)
        {
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

    fn render(&mut self, pc: &VoxelPushConstants) -> Result<(), Box<dyn std::error::Error>> {
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
        let pc_bytes = bytemuck::bytes_of(pc);

        encoder.draw(
            &frame.texture,
            &self.pipeline,
            3,
            pc_bytes,
            [0.1, 0.1, 0.15, 1.0],
        )?;

        encoder.transition_to_present(&frame.texture);
        self.queue.submit_surface(encoder.finish(), frame)?;

        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("  lgpu Voxel Renderer | Real-Time DDA Raymarching Engine");
    println!("  Built on modern Vulkan 1.3 Dynamic Rendering & Bindless");
    println!("============================================================");
    println!("Controls:");
    println!("  [W, A, S, D]       Fly forward / left / back / right");
    println!("  [Space, Shift]     Fly up / down");
    println!("  [Mouse Drag]       Look around (yaw / pitch)");
    println!("  [Arrow Keys]       Look around alternative");
    println!("  [1, 2, 3]          Switch Voxel Worlds (Highlands, Islands, Citadel)");
    println!("  [Tab]              Cycle Render Modes (Beauty, AO, Normals, Depth, IDs)");
    println!("  [T]                Toggle Day/Night Cycle Animation");
    println!("  [R]                Reset Camera");
    println!("  [Esc]              Exit");
    println!("============================================================");

    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("lgpu Voxel Renderer — Real-Time DDA Engine")
            .with_inner_size(LogicalSize::new(1280.0, 720.0))
            .build(&event_loop)?,
    );

    let instance = Instance::new()?;
    let adapters = instance.enumerate_adapters()?;
    if adapters.is_empty() {
        eprintln!("Error: No Vulkan 1.3 physical GPUs found!");
        return Ok(());
    }

    let adapter = &adapters[0];
    let gpu_info = adapter.info();
    println!(
        "GPU: {} (Type: {:?}, Vulkan 1.3 Dynamic Rendering: {})",
        gpu_info.name, gpu_info.device_type, gpu_info.supports_dynamic_rendering
    );

    let (device, queue) = adapter.request_device(&instance)?;
    let surface = instance.create_surface(&window)?;

    let size = window.inner_size();
    let mut renderer = VoxelRenderer::new(device, queue, surface, size.width, size.height)?;

    let mut camera = Camera::new([0.0, 24.0, -18.0], 90.0, -12.0);
    let mut world_mode: u32 = 0;
    let mut render_mode: u32 = 0;
    let mut animate_sun = true;
    let mut sun_angle: f32 = 55.0;

    let mut held_keys: HashSet<winit::keyboard::KeyCode> = HashSet::new();
    let mut mouse_dragging = false;
    let mut last_mouse_pos: Option<(f64, f64)> = None;

    let start_time = Instant::now();
    let mut last_frame = Instant::now();
    let mut frame_count = 0u32;
    let mut last_fps_time = Instant::now();

    event_loop.run(move |event, target| {
        target.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => {
                    target.exit();
                }
                WindowEvent::Resized(physical_size) => {
                    if let Err(e) = renderer.resize(physical_size.width, physical_size.height) {
                        eprintln!("Error resizing surface: {e:?}");
                    }
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key,
                            state,
                            logical_key,
                            ..
                        },
                    ..
                } => {
                    if let winit::keyboard::PhysicalKey::Code(code) = physical_key {
                        match state {
                            ElementState::Pressed => {
                                held_keys.insert(code);
                            }
                            ElementState::Released => {
                                held_keys.remove(&code);
                            }
                        }
                    }

                    if state == ElementState::Pressed {
                        match logical_key {
                            Key::Named(NamedKey::Escape) => {
                                target.exit();
                            }
                            Key::Named(NamedKey::Tab) => {
                                render_mode = (render_mode + 1) % 5;
                                let mode_name = match render_mode {
                                    0 => "Full Beauty (Diffuse + AO + Shadows + Fog)",
                                    1 => "Ambient Occlusion Only (Clay Mode)",
                                    2 => "Surface Normals (RGB)",
                                    3 => "Depth / Distance",
                                    _ => "Voxel Material IDs",
                                };
                                println!("Render Mode -> [{}] {}", render_mode, mode_name);
                            }
                            Key::Character(ref ch) => match ch.as_str() {
                                "1" => {
                                    world_mode = 0;
                                    println!("World -> Mountain Highlands & Lakes");
                                }
                                "2" => {
                                    world_mode = 1;
                                    camera.pos = [0.0, 32.0, -22.0];
                                    println!("World -> Floating Voxel Islands");
                                }
                                "3" => {
                                    world_mode = 2;
                                    camera.pos = [0.0, 18.0, -32.0];
                                    println!("World -> Ancient Voxel Citadel & Ruins");
                                }
                                "t" | "T" => {
                                    animate_sun = !animate_sun;
                                    println!("Sun Animation -> {}", if animate_sun { "ON" } else { "PAUSED" });
                                }
                                "r" | "R" => {
                                    camera.pos = [0.0, 24.0, -18.0];
                                    camera.yaw = 90.0;
                                    camera.pitch = -12.0;
                                    println!("Camera reset");
                                }
                                _ => {}
                            },
                            _ => {}
                        }
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == MouseButton::Left {
                        mouse_dragging = state == ElementState::Pressed;
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if mouse_dragging {
                        if let Some((lx, ly)) = last_mouse_pos {
                            let dx = (position.x - lx) as f32;
                            let dy = (position.y - ly) as f32;
                            camera.yaw += dx * 0.25;
                            camera.pitch = (camera.pitch - dy * 0.25).clamp(-88.0, 88.0);
                        }
                    }
                    last_mouse_pos = Some((position.x, position.y));
                }
                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let dt = now.duration_since(last_frame).as_secs_f32().min(0.1);
                    last_frame = now;

                    // Camera movement based on held keys
                    let fwd = camera.forward_vector();
                    let right = camera.right_vector();
                    let _up = camera.up_vector();
                    let speed = 18.0 * dt;

                    use winit::keyboard::KeyCode;
                    if held_keys.contains(&KeyCode::KeyW) {
                        camera.pos[0] += fwd[0] * speed;
                        camera.pos[1] += fwd[1] * speed;
                        camera.pos[2] += fwd[2] * speed;
                    }
                    if held_keys.contains(&KeyCode::KeyS) {
                        camera.pos[0] -= fwd[0] * speed;
                        camera.pos[1] -= fwd[1] * speed;
                        camera.pos[2] -= fwd[2] * speed;
                    }
                    if held_keys.contains(&KeyCode::KeyD) {
                        camera.pos[0] += right[0] * speed;
                        camera.pos[1] += right[1] * speed;
                        camera.pos[2] += right[2] * speed;
                    }
                    if held_keys.contains(&KeyCode::KeyA) {
                        camera.pos[0] -= right[0] * speed;
                        camera.pos[1] -= right[1] * speed;
                        camera.pos[2] -= right[2] * speed;
                    }
                    if held_keys.contains(&KeyCode::Space) {
                        camera.pos[1] += speed;
                    }
                    if held_keys.contains(&KeyCode::ShiftLeft) || held_keys.contains(&KeyCode::ShiftRight) {
                        camera.pos[1] -= speed;
                    }
                    if held_keys.contains(&KeyCode::ArrowLeft) {
                        camera.yaw -= 60.0 * dt;
                    }
                    if held_keys.contains(&KeyCode::ArrowRight) {
                        camera.yaw += 60.0 * dt;
                    }
                    if held_keys.contains(&KeyCode::ArrowUp) {
                        camera.pitch = (camera.pitch + 45.0 * dt).clamp(-88.0, 88.0);
                    }
                    if held_keys.contains(&KeyCode::ArrowDown) {
                        camera.pitch = (camera.pitch - 45.0 * dt).clamp(-88.0, 88.0);
                    }

                    // Animate day-night sun cycle
                    if animate_sun {
                        sun_angle += 14.0 * dt;
                        if sun_angle > 360.0 {
                            sun_angle -= 360.0;
                        }
                    }

                    let aspect = renderer.width as f32 / renderer.height.max(1) as f32;
                    let elapsed = start_time.elapsed().as_secs_f32();

                    let pc = VoxelPushConstants {
                        cam_pos: camera.pos,
                        time: elapsed,
                        cam_dir: camera.forward_vector(),
                        aspect,
                        cam_up: camera.up_vector(),
                        mode: world_mode,
                        render_mode,
                        sun_angle,
                        fov: camera.fov,
                        _pad0: 0.0,
                        _pad1: 0.0,
                        _pad2: 0.0,
                    };

                    if let Err(e) = renderer.render(&pc) {
                        eprintln!("Render error: {e:?}");
                    }

                    frame_count += 1;
                    if last_fps_time.elapsed().as_secs_f32() >= 1.0 {
                        let fps = frame_count;
                        frame_count = 0;
                        last_fps_time = Instant::now();
                        let title = format!(
                            "lgpu Voxel Renderer | FPS: {} | World: {} | Cam: [{:.1}, {:.1}, {:.1}]",
                            fps,
                            match world_mode {
                                0 => "Highlands",
                                1 => "Islands",
                                _ => "Citadel",
                            },
                            camera.pos[0],
                            camera.pos[1],
                            camera.pos[2]
                        );
                        window.set_title(&title);
                    }

                    window.request_redraw();
                }
                _ => {}
            },
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    })?;

    Ok(())
}
