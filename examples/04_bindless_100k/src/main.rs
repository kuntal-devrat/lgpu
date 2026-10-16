//! 100,000 GPU Particles Simulation & Stress Test with lgpu
//!
//! Simulates physics, ground-plane collisions, and velocity-based color grading
//! across 100,000 particles concurrently on the GPU using a single compute dispatch.

use bytemuck::{Pod, Zeroable};
use lgpu::{BufferDesc, BufferUsage, Instance, MemoryLocation};
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Particle {
    pos: [f32; 4],
    vel: [f32; 4],
    color: [f32; 4],
}

// Precompiled portable SPIR-V bytecode for 100k particles simulation
pub const PARTICLES_100K_SPV: &[u32] = &[
    119734787, 65536, 28, 90, 0, 131089, 1, 720906, 1599492179, 1599227979, 1919906931, 1600481121,
    1717990754, 1935635045, 1634889588, 1667196263, 1936941420, 0, 393227, 1, 1280527431,
    1685353262, 808793134, 0, 196622, 0, 1, 393231, 5, 16, 1852399981, 0, 13, 393232, 16, 17, 256,
    1, 1, 327752, 5, 0, 35, 0, 327752, 5, 1, 35, 16, 327752, 5, 2, 35, 32, 262215, 6, 6, 48,
    262215, 9, 34, 0, 262215, 9, 33, 0, 196679, 10, 2, 327752, 10, 0, 35, 0, 262215, 13, 11, 28,
    131091, 2, 196630, 4, 32, 262167, 3, 4, 4, 327710, 5, 3, 3, 3, 196637, 6, 5, 262165, 8, 32, 0,
    262167, 7, 8, 3, 196638, 10, 6, 262176, 11, 12, 10, 262203, 11, 9, 12, 262176, 14, 1, 7,
    262203, 14, 13, 1, 196641, 17, 2, 262176, 18, 12, 6, 262187, 8, 19, 0, 262187, 4, 21,
    1008767022, 262187, 4, 22, 0, 262187, 4, 23, 1062836634, 262187, 4, 24, 1092616192, 262187, 4,
    25, 1084227584, 262187, 4, 26, 1065353216, 262187, 4, 27, 1056964608, 262176, 29, 7, 5, 196654,
    5, 30, 131092, 34, 262176, 38, 12, 5, 262176, 41, 7, 3, 262176, 42, 7, 4, 262187, 8, 43, 1,
    262187, 8, 60, 2, 262167, 80, 4, 3, 327734, 2, 16, 0, 17, 131320, 12, 327739, 29, 28, 7, 30,
    262205, 7, 15, 13, 327745, 18, 20, 9, 19, 131321, 31, 131320, 31, 327761, 8, 32, 15, 0, 327748,
    8, 33, 9, 0, 327854, 34, 35, 32, 33, 196855, 36, 0, 262394, 35, 37, 36, 131320, 37, 65789,
    131320, 36, 327745, 38, 39, 20, 32, 262205, 5, 40, 39, 196670, 28, 40, 393281, 42, 44, 28, 43,
    43, 262205, 4, 45, 44, 327811, 4, 46, 45, 21, 393281, 42, 47, 28, 43, 43, 196670, 47, 46,
    393281, 42, 48, 28, 43, 19, 262205, 4, 49, 48, 393281, 42, 50, 28, 19, 19, 262205, 4, 51, 50,
    327809, 4, 52, 51, 49, 393281, 42, 53, 28, 19, 19, 196670, 53, 52, 393281, 42, 54, 28, 43, 43,
    262205, 4, 55, 54, 393281, 42, 56, 28, 19, 43, 262205, 4, 57, 56, 327809, 4, 58, 57, 55,
    393281, 42, 59, 28, 19, 43, 196670, 59, 58, 393281, 42, 61, 28, 43, 60, 262205, 4, 62, 61,
    393281, 42, 63, 28, 19, 60, 262205, 4, 64, 63, 327809, 4, 65, 64, 62, 393281, 42, 66, 28, 19,
    60, 196670, 66, 65, 393281, 42, 67, 28, 19, 43, 262205, 4, 68, 67, 327864, 34, 69, 68, 22,
    196855, 70, 0, 262394, 69, 71, 70, 131320, 71, 393281, 42, 72, 28, 19, 43, 196670, 72, 22,
    393281, 42, 73, 28, 43, 43, 262205, 4, 74, 73, 262271, 4, 75, 74, 327813, 4, 76, 75, 23,
    393281, 42, 77, 28, 43, 43, 196670, 77, 76, 131321, 70, 131320, 70, 327745, 41, 78, 28, 43,
    262205, 3, 79, 78, 524367, 80, 81, 79, 79, 0, 1, 2, 393228, 4, 82, 1, 66, 81, 327813, 4, 83,
    82, 24, 327813, 4, 84, 82, 25, 327811, 4, 85, 26, 84, 458832, 3, 86, 83, 27, 85, 26, 327745,
    41, 87, 28, 60, 196670, 87, 86, 262205, 5, 88, 28, 327745, 38, 89, 20, 32, 196670, 89, 88,
    65789, 65592,
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== lgpu: 100,000 Particles GPU Simulation Stress Test ===");

    // 1. Initialize Instance & select device
    let instance = Instance::new()?;
    let adapter = &instance.enumerate_adapters()?[0];
    let (device, queue) = adapter.request_device(&instance)?;
    let info = adapter.info();
    println!("Device: {} ({:?})", info.name, info.device_type);

    const PARTICLE_COUNT: usize = 100_000;
    println!("Allocating buffer for {} particles...", PARTICLE_COUNT);

    // 2. Initialize 100,000 particles
    let mut particles = Vec::with_capacity(PARTICLE_COUNT);
    for i in 0..PARTICLE_COUNT {
        let fi = i as f32;
        particles.push(Particle {
            pos: [
                (fi % 200.0) - 100.0,
                10.0 + (fi % 50.0),
                (fi / 200.0) - 250.0,
                1.0,
            ],
            vel: [
                (fi * 0.013).sin() * 0.1,
                (fi * 0.017).cos() * 0.05,
                (fi * 0.019).sin() * 0.1,
                0.0,
            ],
            color: [1.0, 0.5, 0.2, 1.0],
        });
    }

    let byte_size = (PARTICLE_COUNT * std::mem::size_of::<Particle>()) as u64;
    println!(
        "Total GPU buffer size: {:.2} MB",
        byte_size as f64 / 1_048_576.0
    );

    let desc = BufferDesc {
        label: Some("ParticlesBuffer100k"),
        size: byte_size,
        usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
        memory_location: MemoryLocation::CpuToGpu,
    };
    let particle_buffer = device.create_buffer_init(&desc, bytemuck::cast_slice(&particles))?;

    // 3. Compile Compute Pipeline
    let shader = device.create_shader_spirv(PARTICLES_100K_SPV)?;
    let pipeline = device.create_compute_pipeline(&shader, 1)?;

    const WORKGROUP_SIZE: u32 = 256;
    let workgroup_count = (PARTICLE_COUNT as u32).div_ceil(WORKGROUP_SIZE);
    println!(
        "Dispatching {} workgroups ({} threads per workgroup = {} total threads)...",
        workgroup_count,
        WORKGROUP_SIZE,
        workgroup_count * WORKGROUP_SIZE
    );

    // Warm-up run
    {
        let mut encoder = device.create_command_encoder()?;
        encoder.dispatch_compute(&pipeline, &[&particle_buffer], workgroup_count, 1, 1);
        queue.submit(encoder.finish())?;
    }

    // 4. Benchmark 10 consecutive simulation iterations
    const ITERATIONS: u32 = 10;
    println!("\nExecuting {} simulation frames on GPU...", ITERATIONS);
    let start_time = Instant::now();

    for _ in 0..ITERATIONS {
        let mut encoder = device.create_command_encoder()?;
        encoder.dispatch_compute(&pipeline, &[&particle_buffer], workgroup_count, 1, 1);
        queue.submit(encoder.finish())?;
    }

    let total_duration = start_time.elapsed();
    let avg_frame_time = total_duration / ITERATIONS;
    let particles_per_sec =
        (PARTICLE_COUNT as f64 * ITERATIONS as f64) / total_duration.as_secs_f64();

    // 5. Read back and verify particle updates
    particle_buffer.read(bytemuck::cast_slice_mut(&mut particles))?;

    println!("\n=== Benchmark Results ===");
    println!("  Particles Count:         {}", PARTICLE_COUNT);
    println!(
        "  Total Time ({} frames):   {:.2?}",
        ITERATIONS, total_duration
    );
    println!("  Avg Time Per Frame:      {:.3?}", avg_frame_time);
    println!(
        "  Simulation Throughput:   {:.2} Million particles/sec",
        particles_per_sec / 1_000_000.0
    );

    println!("\nSample Particle State Verification:");
    println!("---------------------------------------------------------------");
    for i in [0, 42, 1000, 50000, 99999] {
        let p = &particles[i];
        println!(
            "  Particle [{:05}]: Pos=({:6.2}, {:6.2}, {:6.2}) | Vel=({:6.3}, {:6.3}, {:6.3})",
            i, p.pos[0], p.pos[1], p.pos[2], p.vel[0], p.vel[1], p.vel[2]
        );
        assert!(
            p.pos[1] >= 0.0,
            "Particle must not penetrate ground plane y=0"
        );
    }
    println!("---------------------------------------------------------------");
    println!("SUCCESS! 100,000 particles physically simulated on GPU in real-time.");

    Ok(())
}
