//! Integration test verifying dynamic Vulkan loading and adapter enumeration.

use lgpu::Instance;

#[test]
fn test_vulkan_instance_and_adapters() {
    let instance = Instance::new().expect("Failed to initialize lgpu Vulkan instance");
    let adapters = instance
        .enumerate_adapters()
        .expect("Failed to enumerate adapters");

    println!("Discovered {} GPU adapter(s):", adapters.len());
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

    assert!(!adapters.is_empty(), "Expected at least one Vulkan adapter");

    let (device, queue) = adapters[0]
        .request_device(&instance)
        .expect("Failed to request logical device");

    println!("Successfully initialized logical device and queue!");
}
