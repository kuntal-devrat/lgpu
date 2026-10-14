
#![allow(unused)]
use ash::vk;
fn check() {
    let app = vk::ApplicationInfo::default();
    let _ = app.application_name;
}
