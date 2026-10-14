//! Modern Dynamic Rendering Triangle rasterization example with lgpu
//!
//! Renders a procedural RGB triangle to an offscreen texture using Vulkan 1.3
//! dynamic rendering with zero vertex buffers (coordinates derived dynamically
//! in shader), and renders an ASCII visualization in the terminal.

use lgpu::{
    BufferDesc, BufferUsage, Instance, MemoryLocation, TextureDesc, TextureFormat, TextureUsage,
};

// Precompiled portable SPIR-V bytecode containing both vs_main and fs_main
pub const TRIANGLE_SPV: &[u32] = &[
    119734787, 65536, 28, 67, 0, 131089, 1, 393227, 1, 1280527431, 1685353262, 808793134, 0,
    196622, 0, 1, 524303, 0, 16, 1834972022, 7235937, 9, 12, 14, 524303, 4, 63, 1834972006,
    7235937, 56, 59, 62, 196624, 63, 7, 327752, 6, 0, 35, 0, 327752, 6, 1, 35, 16, 262215,
    9, 11, 42, 262215, 12, 11, 0, 262215, 14, 30, 0, 262215, 56, 11, 15, 262215, 59, 30, 0,
    262215, 62, 30, 0, 131091, 2, 196630, 4, 32, 262167, 3, 4, 4, 262167, 5, 4, 3, 262174, 6,
    3, 5, 262165, 7, 32, 0, 262176, 10, 1, 7, 262203, 10, 9, 1, 262176, 13, 3, 3, 262203, 13,
    12, 3, 262176, 15, 3, 5, 262203, 15, 14, 3, 196641, 17, 2, 262187, 7, 18, 0, 262187, 4,
    19, 0, 262187, 4, 20, 3204448256, 262187, 4, 21, 1065353216, 458796, 3, 22, 19, 20, 19,
    21, 393260, 5, 23, 21, 19, 19, 262187, 7, 24, 1, 262187, 4, 25, 1056964608, 458796, 3,
    26, 25, 25, 19, 21, 393260, 5, 27, 19, 21, 19, 458796, 3, 28, 20, 25, 19, 21, 393260, 5,
    29, 19, 19, 21, 262176, 31, 7, 6, 196654, 6, 32, 131092, 34, 262176, 39, 7, 3, 262176,
    41, 7, 5, 262176, 57, 1, 3, 262203, 57, 56, 1, 262176, 60, 1, 5, 262203, 60, 59, 1,
    262203, 13, 62, 3, 327734, 2, 16, 0, 17, 131320, 8, 327739, 31, 30, 7, 32, 262205, 7,
    11, 9, 131321, 33, 131320, 33, 327850, 34, 35, 11, 18, 196855, 36, 0, 262394, 35, 37,
    38, 131320, 37, 327745, 39, 40, 30, 18, 196670, 40, 22, 327745, 41, 42, 30, 24, 196670,
    42, 23, 131321, 36, 131320, 38, 327850, 34, 43, 11, 24, 196855, 44, 0, 262394, 43, 45,
    46, 131320, 45, 327745, 39, 47, 30, 18, 196670, 47, 26, 327745, 41, 48, 30, 24, 196670,
    48, 27, 131321, 44, 131320, 46, 327745, 39, 49, 30, 18, 196670, 49, 28, 327745, 41, 50,
    30, 24, 196670, 50, 29, 131321, 44, 131320, 44, 131321, 36, 131320, 36, 262205, 6, 51,
    30, 327761, 3, 52, 51, 0, 196670, 12, 52, 327761, 5, 53, 51, 1, 196670, 14, 53, 65789,
    65592, 327734, 2, 63, 0, 17, 131320, 54, 262205, 3, 58, 56, 262205, 5, 61, 59, 327760,
    6, 55, 58, 61, 131321, 64, 131320, 64, 327761, 5, 65, 55, 1, 327760, 3, 66, 65, 21,
    196670, 62, 66, 65789, 65592,
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== lgpu: Modern Dynamic Rendering Triangle ===");

    // 1. Initialize Instance
    let instance = Instance::new()?;
    let adapter = &instance.enumerate_adapters()?[0];
    let (device, queue) = adapter.request_device(&instance)?;
    let info = adapter.info();
    println!("Using GPU: {} ({:?})", info.name, info.device_type);

    const WIDTH: u32 = 40;
    const HEIGHT: u32 = 24;
    let num_pixels = (WIDTH * HEIGHT) as usize;

    // 2. Create Render Target Texture & Readback Buffer
    let texture_desc = TextureDesc {
        label: Some("TriangleTarget"),
        width: WIDTH,
        height: HEIGHT,
        depth: 1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    };
    let render_target = device.create_texture(&texture_desc)?;

    let buffer_desc = BufferDesc {
        label: Some("TriangleReadback"),
        size: (num_pixels * 4) as u64,
        usage: BufferUsage::TRANSFER_DST,
        memory_location: MemoryLocation::GpuToCpu,
    };
    let readback_buf = device.create_buffer(&buffer_desc)?;

    // 3. Create Graphics Pipeline with Dynamic Rendering
    let shader = device.create_shader_spirv(TRIANGLE_SPV)?;
    let pipeline = device.create_graphics_pipeline(
        &shader,
        "vs_main",
        "fs_main",
        TextureFormat::Rgba8Unorm,
    )?;

    // 4. Render Triangle (dark background [0.0, 0.0, 0.0, 1.0])
    let mut encoder = device.create_command_encoder()?;
    encoder.render_triangle(&render_target, &pipeline, [0.0, 0.0, 0.0, 1.0]);
    encoder.copy_texture_to_buffer(&render_target, &readback_buf);
    queue.submit(encoder.finish())?;

    // 5. Read back pixels
    let mut pixels = vec![0u8; num_pixels * 4];
    readback_buf.read(&mut pixels)?;

    // 6. Terminal ASCII Visualization of GPU-Rasterized Triangle
    println!("\nGPU Rasterized Output Preview ({}x{} pixels):", WIDTH, HEIGHT);
    println!("┌{}┐", "─".repeat(WIDTH as usize));

    let mut rendered_triangle_pixels = 0;
    for y in 0..HEIGHT {
        print!("│");
        for x in 0..WIDTH {
            let idx = ((y * WIDTH + x) * 4) as usize;
            let r = pixels[idx + 0];
            let g = pixels[idx + 1];
            let b = pixels[idx + 2];

            if r > 20 || g > 20 || b > 20 {
                rendered_triangle_pixels += 1;
                // Terminal ANSI colored block
                print!("\x1b[38;2;{};{};{}m▲\x1b[0m", r, g, b);
            } else {
                print!(" ");
            }
        }
        println!("│");
    }
    println!("└{}┘", "─".repeat(WIDTH as usize));

    println!(
        "\nSUCCESS! Triangle rasterized on GPU: {} pixels shaded.",
        rendered_triangle_pixels
    );
    assert!(
        rendered_triangle_pixels > 50,
        "Expected GPU to rasterize triangle pixels"
    );

    Ok(())
}
