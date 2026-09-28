use spirv_builder::{SpirvBuilder, SpirvMetadata};
use std::{env, fs, path::PathBuf};

fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    let shader_crate = format!("{manifest_dir}/../skybound_gpu");

    let out_path = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo should set OUT_DIR"))
        .join("skybound_gpu.spv");

    // Tell Cargo when to rebuild
    println!("cargo:rerun-if-changed={shader_crate}");

    let result = SpirvBuilder::new(shader_crate, "spirv-unknown-vulkan1.1")
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
