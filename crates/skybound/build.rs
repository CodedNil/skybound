use spirv_builder::{SpirvBuilder, SpirvMetadata};
use std::path::PathBuf;

fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    // Path to the shader crate relative to this crate
    let shader_project_path = PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .join("shaders")
        .join("raymarch");

    // Path to the assets directory in the root of the project
    let shader_output_path = PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets")
        .join("shaders")
        .join("raymarch.spv");

    println!(
        "cargo:rerun-if-changed={}",
        shader_project_path.to_str().unwrap()
    );

    let result = SpirvBuilder::new(shader_project_path, "spirv-unknown-vulkan1.1")
        .spirv_metadata(SpirvMetadata::None)
        .release(true)
        .uniform_buffer_standard_layout(true)
        .relax_block_layout(true)
        .scalar_block_layout(true)
        .build()
        .expect("Failed to build rust-gpu shader");

    // Copy the built shader to the assets directory
    let built_shader_path = result.module.unwrap_single();
    std::fs::create_dir_all(shader_output_path.parent().unwrap()).ok();
    std::fs::copy(built_shader_path, shader_output_path)
        .expect("Failed to copy built shader to assets");
}
