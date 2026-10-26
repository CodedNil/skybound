use spirv_builder::{SpirvBuilder, SpirvMetadata};
use std::{env, fs, path::PathBuf, thread};

/// Build the shader crate to SPIR-V with `features`, in its own target folder so two builds can run
/// at once, and copy the module to `name` in Cargo's build output.
fn build(manifest_dir: &str, features: &[&str], folder: &str, name: &str) {
    let out_path =
        PathBuf::from(env::var_os("OUT_DIR").expect("Cargo should set OUT_DIR")).join(name);
    let result = SpirvBuilder::new(manifest_dir, "spirv-unknown-vulkan1.1")
        .shader_crate_default_features(false)
        .shader_crate_features(features.iter().map(ToString::to_string))
        .target_dir_path(folder)
        .spirv_metadata(SpirvMetadata::None)
        .release(true)
        .uniform_buffer_standard_layout(true)
        .relax_block_layout(true)
        .scalar_block_layout(true)
        .build()
        .expect("Failed to build rust-gpu shader");
    let built_shader = result.module.unwrap_single();
    fs::copy(built_shader, out_path).expect("Failed to copy shader to Cargo's build output");
}

fn main() {
    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("spirv") {
        return;
    }
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    println!("cargo:rerun-if-changed=src/scene");
    // The game's passes and the previewer's tracer are two modules, built side by side: each
    // inlines the whole creature tracer, and one module holding both takes several times as long.
    let preview = thread::spawn(move || {
        build(
            manifest_dir,
            &["preview"],
            "spirv-builder-preview",
            "preview.spv",
        );
    });
    build(manifest_dir, &[], "spirv-builder", "scene.spv");
    preview.join().expect("preview shader build");
}
