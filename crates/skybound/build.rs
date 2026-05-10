use spirv_builder::{SpirvBuilder, SpirvMetadata};
use std::fs;

fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    let shader_crate = format!("{manifest_dir}/../shaders/raymarch");
    let shared_crate = format!("{manifest_dir}/../skybound_shared");
    let dest_path = format!("{manifest_dir}/../../assets/shaders/raymarch.spv");

    // Tell Cargo when to rebuild
    println!("cargo:rerun-if-changed={shader_crate}");
    println!("cargo:rerun-if-changed={shared_crate}");

    let result = SpirvBuilder::new(shader_crate, "spirv-unknown-vulkan1.1")
        .spirv_metadata(SpirvMetadata::None)
        .release(true)
        .uniform_buffer_standard_layout(true)
        .relax_block_layout(true)
        .scalar_block_layout(true)
        .build()
        .expect("Failed to build rust-gpu shader");

    // Copy the built shader
    let built_shader = result.module.unwrap_single();
    if let Some(parent) = std::path::Path::new(&dest_path).parent() {
        fs::create_dir_all(parent).ok();
    }
    fs::copy(built_shader, dest_path).expect("Failed to copy shader to assets");
}
