#![no_std]

mod interface;
pub use interface::{PLANET_RADIUS, ShipUniform, ViewUniform};

#[cfg(target_arch = "spirv")]
pub mod shader;
