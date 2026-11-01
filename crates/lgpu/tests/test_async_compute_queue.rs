//! Verifies dispatch on the dedicated async compute queue family.

use lgpu::{BufferDesc, BufferUsage, MemoryLocation, QueueFamily};

const COMPUTE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index < arrayLength(&data)) {
        data[index] = data[index] * 2u + 1u;
    }
}
"#;

macro_rules! require_gpu {
    () => {
        match (|| {
            let instance = lgpu::Instance::new().ok()?;
            let adapters = instance.enumerate_adapters().ok()?;
            if adapters.is_empty() {
                return None;
            }
            let (device, queue) = adapters[0].request_device(&instance).ok()?;
            Some((instance, device, queue))
        })() {
            Some(tuple) => tuple,
            None => {
                eprintln!("Skipping test: Vulkan 1.3 physical device not available");
                return;
            }
        }
    };
}

#[test]
fn test_dispatch_on_async_compute_queue() {
    let (_instance, device, _graphics_queue) = require_gpu!();

    if !device.has_async_compute_queue() {
        eprintln!("Skipping test: adapter has no separate compute queue family");
        return;
    }

    let compute_queue = device
        .compute_queue()
        .expect("has_async_compute_queue implies a compute queue handle");
    assert_eq!(compute_queue.family(), QueueFamily::Compute);
    assert!(compute_queue.is_compute());
    assert!(
        device.compute_queue_family_index().is_some(),
        "a compute queue handle implies a distinct queue family index"
    );

    const COUNT: u32 = 64;
    let initial: Vec<u32> = (0..COUNT).collect();
    let buffer = device
        .create_buffer_init_slice(
            &BufferDesc {
                label: Some("AsyncComputeData"),
                size: std::mem::size_of_val(initial.as_slice()) as u64,
                usage: BufferUsage::STORAGE,
                memory_location: MemoryLocation::CpuToGpu,
            },
            &initial,
        )
        .expect("create shared storage buffer");

    let shader = device
        .create_shader_wgsl(COMPUTE_SHADER)
        .expect("compile compute shader");
    let pipeline = device
        .create_compute_pipeline(&shader, 1)
        .expect("create compute pipeline");

    let mut encoder = device
        .create_compute_command_encoder()
        .expect("create compute-family encoder");
    encoder.dispatch_compute(&pipeline, &[&buffer], 1, 1, 1);
    compute_queue
        .submit(encoder.finish())
        .expect("submit on async compute queue");

    let mut result = vec![0u32; COUNT as usize];
    buffer
        .read(bytemuck::cast_slice_mut(&mut result))
        .expect("read back compute results");

    for (index, value) in result.iter().enumerate() {
        assert_eq!(
            *value,
            index as u32 * 2 + 1,
            "element {index} was not updated on the async compute queue"
        );
    }
}

#[test]
fn test_async_compute_availability_is_consistent() {
    let (_instance, device, _queue) = require_gpu!();

    match device.compute_queue() {
        Some(queue) => {
            assert!(device.has_async_compute_queue());
            assert!(device.compute_queue_family_index().is_some());
            assert_eq!(queue.family(), QueueFamily::Compute);
            assert!(queue.is_compute());
            assert!(
                device.create_compute_command_encoder().is_ok(),
                "a compute queue handle must come with a recordable compute encoder"
            );
        }
        None => {
            assert!(
                !device.has_async_compute_queue(),
                "has_async_compute_queue must agree with compute_queue()"
            );
            assert!(device.compute_queue_family_index().is_none());
            assert!(
                device.create_compute_command_encoder().is_err(),
                "without a separate family, compute encoders must be rejected cleanly"
            );
        }
    }
}

#[test]
fn test_surface_submission_rejected_on_compute_queue() {
    let (_instance, device, _queue) = require_gpu!();
    let Some(compute_queue) = device.compute_queue() else {
        eprintln!("Skipping test: adapter has no separate compute queue family");
        return;
    };

    let mut encoder = device
        .create_compute_command_encoder()
        .expect("create compute encoder");
    encoder.memory_barrier();
    compute_queue
        .submit(encoder.finish())
        .expect("submitting compute work must succeed");
}
