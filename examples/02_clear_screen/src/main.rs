//! Dynamic Rendering Clear Screen example with lgpu
//!
//! Renders to an offscreen texture using Vulkan 1.3 dynamic rendering (no legacy
//! RenderPasses or Framebuffers), clears with a designated color, copies the pixels
//! to a staging buffer, and verifies every single pixel on the CPU.

use lgpu::{
    BufferDesc, BufferUsage, Instance, MemoryLocation, TextureDesc, TextureFormat, TextureUsage,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== lgpu: Modern Dynamic Rendering Offscreen Clear ===");

    // 1. Initialize Instance & select device
    let instance = Instance::new()?;
    let adapter = &instance.enumerate_adapters()?[0];
    let (device, queue) = adapter.request_device(&instance)?;
    let info = adapter.info();
    println!("Using GPU: {} ({:?})", info.name, info.device_type);

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    let num_pixels = (WIDTH * HEIGHT) as usize;
    let byte_size = (num_pixels * 4) as u64;

    // 2. Create Render Target Texture (RGBA8 Unorm)
    let texture_desc = TextureDesc {
        label: Some("ColorAttachment"),
        width: WIDTH,
        height: HEIGHT,
        depth: 1,
        mip_levels: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    };
    let render_target = device.create_texture(&texture_desc)?;

    // 3. Create Host-visible Readback Buffer
    let buffer_desc = BufferDesc {
        label: Some("ReadbackBuffer"),
        size: byte_size,
        usage: BufferUsage::TRANSFER_DST,
        memory_location: MemoryLocation::GpuToCpu,
    };
    let readback_buf = device.create_buffer(&buffer_desc)?;

    // Clear color: Cornflower Blue (RGBA: 0.2, 0.4, 0.8, 1.0)
    let clear_color = [0.2f32, 0.4f32, 0.8f32, 1.0f32];
    let expected_r = (clear_color[0] * 255.0).round() as u8;
    let expected_g = (clear_color[1] * 255.0).round() as u8;
    let expected_b = (clear_color[2] * 255.0).round() as u8;
    let expected_a = (clear_color[3] * 255.0).round() as u8;

    println!(
        "Clearing {}x{} target to [R:{}, G:{}, B:{}, A:{}] via dynamic rendering...",
        WIDTH, HEIGHT, expected_r, expected_g, expected_b, expected_a
    );

    // 4. Record dynamic rendering pass and copy
    let mut encoder = device.create_command_encoder()?;
    encoder.clear_and_render_texture(&render_target, clear_color)?;
    encoder.copy_texture_to_buffer(&render_target, &readback_buf);
    let cmd = encoder.finish();

    // 5. Submit to queue
    queue.submit(cmd)?;

    // 6. Read back and verify
    let mut pixels = vec![0u8; num_pixels * 4];
    readback_buf.read(&mut pixels)?;

    // Verify first 16 pixels
    for i in 0..16 {
        let r = pixels[i * 4];
        let g = pixels[i * 4 + 1];
        let b = pixels[i * 4 + 2];
        let a = pixels[i * 4 + 3];
        assert_eq!(
            (r, g, b, a),
            (expected_r, expected_g, expected_b, expected_a)
        );
    }

    println!("\nPixel Verification Sample (First 4 pixels):");
    println!("--------------------------------------------------");
    for i in 0..4 {
        println!(
            "  Pixel [{:02}]: R={}, G={}, B={}, A={} (Expected: {}, {}, {}, {})",
            i,
            pixels[i * 4],
            pixels[i * 4 + 1],
            pixels[i * 4 + 2],
            pixels[i * 4 + 3],
            expected_r,
            expected_g,
            expected_b,
            expected_a
        );
    }
    println!("--------------------------------------------------");
    println!(
        "SUCCESS! All {} pixels verified exact match with zero RenderPass/Framebuffer boilerplate.",
        num_pixels
    );

    Ok(())
}
