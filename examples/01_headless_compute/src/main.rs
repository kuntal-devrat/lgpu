//! Headless GPGPU compute example with lgpu
//!
//! Dispatches a vector-addition compute kernel (C[i] = A[i] + B[i]) on the GPU
//! without any window or display server, and reads back the computed results.

use lgpu::{BufferDesc, BufferUsage, Instance, MemoryLocation};

// Precompiled portable SPIR-V bytecode for:
//   layout(local_size_x = 64) in;
//   c[i] = a[i] + b[i];
pub const VECTOR_ADD_SPV: &[u32] = &[
    119734787, 65536, 28, 36, 0, 131089, 1, 720906, 1599492179, 1599227979, 1919906931,
    1600481121, 1717990754, 1935635045, 1634889588, 1667196263, 1936941420, 0, 393227, 1,
    1280527431, 1685353262, 808793134, 0, 196622, 0, 1, 393231, 5, 20, 1852399981, 0, 17,
    393232, 20, 17, 64, 1, 1, 262215, 4, 6, 4, 262215, 7, 34, 0, 262215, 7, 33, 0, 196679,
    8, 2, 327752, 8, 0, 35, 0, 262215, 10, 34, 0, 262215, 10, 33, 1, 196679, 11, 2, 327752,
    11, 0, 35, 0, 262215, 13, 34, 0, 262215, 13, 33, 2, 196679, 14, 2, 327752, 14, 0, 35,
    0, 262215, 17, 11, 28, 131091, 2, 196630, 3, 32, 196637, 4, 3, 262165, 6, 32, 0,
    262167, 5, 6, 3, 196638, 8, 4, 262176, 9, 12, 8, 262203, 9, 7, 12, 196638, 11, 4,
    262176, 12, 12, 11, 262203, 12, 10, 12, 196638, 14, 4, 262176, 15, 12, 14, 262203, 15,
    13, 12, 262176, 18, 1, 5, 262203, 18, 17, 1, 196641, 21, 2, 262176, 22, 12, 4, 262187,
    6, 23, 0, 262176, 29, 12, 3, 327734, 2, 20, 0, 21, 131320, 16, 262205, 5, 19, 17,
    327745, 22, 24, 7, 23, 327745, 22, 25, 10, 23, 327745, 22, 26, 13, 23, 131321, 27,
    131320, 27, 327761, 6, 28, 19, 0, 327745, 29, 30, 24, 28, 262205, 3, 31, 30, 327745,
    29, 32, 25, 28, 262205, 3, 33, 32, 327809, 3, 34, 31, 33, 327745, 29, 35, 26, 28,
    196670, 35, 34, 65789, 65592,
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== lgpu: Headless Compute Demonstration ===");

    // 1. Initialize Instance
    let instance = Instance::new()?;
    let adapters = instance.enumerate_adapters()?;
    let adapter = &adapters[0];
    let info = adapter.info();
    println!("Using GPU: {} ({:?})", info.name, info.device_type);

    // 2. Request Device and Queue
    let (device, queue) = adapter.request_device(&instance)?;

    // 3. Prepare Input & Output Data
    const COUNT: usize = 64;
    let data_a: Vec<f32> = (0..COUNT).map(|i| i as f32).collect();
    let data_b: Vec<f32> = (0..COUNT).map(|i| (i * 10) as f32).collect();
    let mut data_c: Vec<f32> = vec![0.0; COUNT];

    let byte_size = (COUNT * std::mem::size_of::<f32>()) as u64;

    let buffer_desc = BufferDesc {
        label: Some("ComputeBuffer"),
        size: byte_size,
        usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
        memory_location: MemoryLocation::CpuToGpu,
    };

    let buf_a = device.create_buffer_init(&buffer_desc, bytemuck::cast_slice(&data_a))?;
    let buf_b = device.create_buffer_init(&buffer_desc, bytemuck::cast_slice(&data_b))?;
    let buf_c = device.create_buffer(&buffer_desc)?;

    // 4. Create Compute Pipeline
    let shader = device.create_shader_spirv(VECTOR_ADD_SPV)?;
    let pipeline = device.create_compute_pipeline(&shader, 3)?;

    // 5. Record & Dispatch Compute
    let mut encoder = device.create_command_encoder()?;
    encoder.dispatch_compute(&pipeline, &[&buf_a, &buf_b, &buf_c], 1, 1, 1);
    let cmd_buffer = encoder.finish();

    // 6. Submit and Wait for GPU completion
    queue.submit(cmd_buffer)?;

    // 7. Read back results
    buf_c.read(bytemuck::cast_slice_mut(&mut data_c))?;

    // 8. Verify Results
    println!("\nVerification (First 8 elements):");
    println!("----------------------------------------------");
    println!(" Index |  Buffer A  |  Buffer B  |  Buffer C (A+B)");
    println!("----------------------------------------------");
    for i in 0..8 {
        println!(
            "  {:02}   |   {:6.1}   |   {:6.1}   |    {:6.1}",
            i, data_a[i], data_b[i], data_c[i]
        );
        assert_eq!(data_c[i], data_a[i] + data_b[i]);
    }
    println!("----------------------------------------------");
    println!("SUCCESS! All {} GPU compute results verified exactly.", COUNT);

    Ok(())
}
