//! Runtime WGSL to SPIR-V compilation using Naga.

use naga::back::spv;
use naga::front::wgsl;

pub fn compile_wgsl_to_spirv(source: &str) -> Result<Vec<u32>, String> {
    let mut frontend = wgsl::Frontend::new();
    let module = frontend
        .parse(source)
        .map_err(|e| format!("WGSL parse error: {e:?}"))?;

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    let info = validator
        .validate(&module)
        .map_err(|e| format!("WGSL validation error: {e:?}"))?;

    let options = spv::Options {
        lang_version: (1, 3),
        ..Default::default()
    };

    let words = spv::write_vec(&module, &info, &options, None)
        .map_err(|e| format!("SPIR-V generation error: {e:?}"))?;

    Ok(words)
}
