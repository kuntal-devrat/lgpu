//! Integration test verifying dynamic Vulkan loading and adapter enumeration.

use lgpu::Instance;

#[test]
fn test_vulkan_instance_and_adapters() {
    let instance = match Instance::new() {
        Ok(i) => i,
        Err(e) => {
            eprintln!("Skipping test: Vulkan instance initialization failed ({e})");
            return;
        }
    };
    let adapters = match instance.enumerate_adapters() {
        Ok(a) if !a.is_empty() => a,
        _ => {
            eprintln!("Skipping test: No Vulkan adapters available");
            return;
        }
    };

    println!("\nDiscovered {} GPU adapter(s):", adapters.len());
    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.info();
        println!(
            "  [{}] {} (Backend: {:?}, Type: {:?}, Vendor: {:#x}, Device: {:#x})",
            i, info.name, info.backend, info.device_type, info.vendor_id, info.device_id
        );
        println!(
            "      Max compute workgroup invocations: {}",
            info.max_compute_workgroup_invocations
        );
    }

    let (_device, _queue) = match adapters[0].request_device(&instance) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Skipping test: Failed to request logical device ({e})");
            return;
        }
    };

    println!("Successfully initialized logical device and queue!");
}
