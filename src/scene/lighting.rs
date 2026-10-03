use core::f32::consts::PI;
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = (1.0 + g2 - 2.0 * g * cos_theta).powf(1.5);
    (1.0 - g2) / (4.0 * PI * denom.max(1e-4))
}
