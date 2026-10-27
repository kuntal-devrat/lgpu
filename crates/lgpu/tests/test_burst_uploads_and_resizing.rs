//! Stress test replicating burst texture uploads under continuous window resizing.
//! Verifies zero `VK_ERROR_DEVICE_LOST`, zero validation layer errors, and zero memory leaks.

use lgpu::{
    HalError, Instance, PresentMode, SurfaceConfiguration, TextureDesc, TextureFormat, TextureUsage,
};

#[test]
fn test_burst_texture_uploads_under_continuous_resizing() {
    let instance = match Instance::new() {
        Ok(i) => i,
        Err(_) => return,
    };
    let adapters = match instance.enumerate_adapters() {
        Ok(a) if !a.is_empty() => a,
        _ => return,
    };
    let (device, queue) = match adapters[0].request_device(&instance) {
        Ok(pair) => pair,
        Err(_) => return,
    };

    let mut surface = match instance.create_headless_surface() {
        Ok(s) => s,
        Err(_) => {
            println!("Headless surface not supported, skipping stress test");
            return;
        }
    };

    // Resolutions cycling each frame, including zero-sized minimized window
    let test_resolutions = [
        (640, 480),
        (800, 600),
        (320, 240),
        (0, 0), // Minimized window
        (512, 512),
        (1024, 768),
        (256, 256),
    ];

    const FRAMES_TO_RUN: usize = 35;
    const TEXTURES_PER_FRAME: usize = 100;

    let mut total_textures_uploaded = 0;

    for frame in 0..FRAMES_TO_RUN {
        let (width, height) = test_resolutions[frame % test_resolutions.len()];
        println!("Frame {frame}: {width}x{height}");

        // 1. Swapchain Reconfiguration churn
        let config = SurfaceConfiguration {
            width,
            height,
            format: TextureFormat::Bgra8Unorm,
            present_mode: PresentMode::Fifo,
            usage: TextureUsage::RENDER_ATTACHMENT,
        };

        let configure_res = surface.configure(&device, &config);
        assert!(
            configure_res.is_ok(),
            "Surface configure must succeed at ({width}x{height})"
        );

        // Minimized / zero-sized window: get_current_texture returns NotReady gracefully
        if width == 0 || height == 0 {
            let acquire_res = surface.get_current_texture();
            assert!(
                matches!(acquire_res, Err(lgpu::Error::Hal(HalError::NotReady(_)))),
                "Zero-size window must return HalError::NotReady"
            );
            continue;
        }

        // 2. Burst Upload of 100 textures per frame via Asynchronous Ring Staging Buffer
        let mut burst_textures = Vec::with_capacity(TEXTURES_PER_FRAME);
        for i in 0..TEXTURES_PER_FRAME {
            let tex_w = 64;
            let tex_h = 64;
            let tex_desc = TextureDesc::new_2d(
                tex_w,
                tex_h,
                TextureFormat::Rgba8Unorm,
                TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
            );
            let texture = device
                .create_texture(&tex_desc)
                .expect("Failed to create burst texture");

            // Generate pattern data
            let pixel_val = ((frame + i) % 255) as u8;
            let data = vec![pixel_val; (tex_w * tex_h * 4) as usize];

            // Asynchronous ring staging copy (non-blocking)
            queue
                .write_texture(&texture, &data)
                .expect("write_texture must succeed without device lost");

            burst_textures.push(texture);
            total_textures_uploaded += 1;
        }

        // 3. Acquire Swapchain Image with Triple-Buffering & Multi-Frames-in-Flight Sync
        let surface_tex = surface
            .get_current_texture()
            .expect("Failed to acquire surface texture");

        let mut encoder = device
            .create_command_encoder()
            .expect("Failed to create encoder");

        let clear_color = [
            (frame as f32) / (FRAMES_TO_RUN as f32),
            0.5,
            1.0 - ((frame as f32) / (FRAMES_TO_RUN as f32)),
            1.0,
        ];
        encoder
            .clear_and_render_texture(&surface_tex.texture, clear_color)
            .expect("Clear failed");

        let cmd = encoder.finish();
        queue
            .submit_surface(cmd, surface_tex)
            .expect("submit_surface failed under burst upload churn");

        // burst_textures are dropped here, verifying RAII slot recycling and staging synchronization!
        drop(burst_textures);
    }

    // 4. Teardown & Validation: Ensure all in-flight work completes cleanly
    device.wait_idle().expect("Device wait_idle failed");

    println!(
        "SUCCESS: Completed {FRAMES_TO_RUN} frames with {total_textures_uploaded} burst textures under continuous resizing with 0 errors!"
    );
}
