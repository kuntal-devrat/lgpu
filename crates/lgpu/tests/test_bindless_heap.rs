//! Integration test verifying the Global Bindless Descriptor Heap.

use lgpu::{BufferDesc, BufferUsage, Instance, MemoryLocation};

#[test]
fn test_bindless_heap_slot_allocation_and_recycling() {
    let instance = Instance::new().expect("Failed to initialize lgpu");
    let adapters = instance.enumerate_adapters().expect("Failed to enumerate adapters");
    assert!(!adapters.is_empty(), "No adapter found");

    let (device, _queue) = adapters[0]
        .request_device(&instance)
        .expect("Failed to create device");

    let desc = BufferDesc {
        label: Some("BindlessTestBuffer"),
        size: 1024,
        usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
        memory_location: MemoryLocation::CpuToGpu,
    };

    println!("\n=== Testing Global Bindless Heap Resource Allocation ===");

    // 1. Allocate 10 buffers and ensure unique slot IDs
    let mut buffers = Vec::new();
    for i in 0..10 {
        let buf = device.create_buffer(&desc).expect("Failed to create buffer");
        let id = buf.id().expect("Buffer should have a bindless ID");
        println!("  Allocated buffer {} -> Bindless Slot {}", i, id.0);
        assert_eq!(id.0, i as u32);
        buffers.push(buf);
    }

    // 2. Drop slots 3 and 7
    println!("  Dropping buffer at slot 3 and slot 7...");
    let dropped_id_3 = buffers.remove(3).id().unwrap();
    let dropped_id_7 = buffers.remove(6).id().unwrap(); // after removing index 3, index 6 was slot 7
    assert_eq!(dropped_id_3.0, 3);
    assert_eq!(dropped_id_7.0, 7);

    // 3. Allocate 2 new buffers: verify slot recycling
    let recycled_1 = device.create_buffer(&desc).expect("Failed to create recycled buffer 1");
    let recycled_2 = device.create_buffer(&desc).expect("Failed to create recycled buffer 2");

    let rec_id_1 = recycled_1.id().unwrap();
    let rec_id_2 = recycled_2.id().unwrap();

    println!("  Recycled Slot A: {}", rec_id_1.0);
    println!("  Recycled Slot B: {}", rec_id_2.0);

    // Slots recycled must be either 7 or 3 (LIFO order from free stack)
    assert!(rec_id_1.0 == 3 || rec_id_1.0 == 7);
    assert!(rec_id_2.0 == 3 || rec_id_2.0 == 7);
    assert_ne!(rec_id_1, rec_id_2);

    println!("SUCCESS! Bindless descriptor slot allocation and recycling verified.");
}
