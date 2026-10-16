//! Integration tests for native Surface configuration and presentation types.

use lgpu::{PresentMode, SurfaceConfiguration, TextureFormat, TextureUsage};

#[test]
fn test_surface_configuration_and_types() {
    let config = SurfaceConfiguration {
        usage: TextureUsage::RENDER_ATTACHMENT,
        format: TextureFormat::Bgra8Unorm,
        width: 1920,
        height: 1080,
        present_mode: PresentMode::Mailbox,
    };

    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
    assert_eq!(config.format, TextureFormat::Bgra8Unorm);
    assert_eq!(config.present_mode, PresentMode::Mailbox);
}
