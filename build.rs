use spirv_builder::{SpirvBuilder, SpirvMetadata};
use std::{env, fs, path::PathBuf};

fn main() {
    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("spirv") {
        return;
    }
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    let out_path =
        PathBuf::from(env::var_os("OUT_DIR").expect("Cargo should set OUT_DIR")).join("scene.spv");

    println!("cargo:rerun-if-changed=src/scene");

    let result = SpirvBuilder::new(manifest_dir, "spirv-unknown-vulkan1.1")
        .shader_crate_default_features(false)
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
