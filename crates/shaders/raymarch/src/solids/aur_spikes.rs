use crate::utils::Textures;
use spirv_std::glam::{FloatExt, Vec2, Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const MAT_SPIKE: u32 = 2;
pub const SPIKE_COLOR: Vec3 = vec3(0.16, 0.09, 0.03);

const SPIKE_BASE_ALT: f32 = -6000.0;
const BASE_UNDULATION: f32 = 1500.0;
const SPIKE_HEIGHT: f32 = 4500.0;

const SPIKE_SCALE: f32 = 0.00003;
const SPIKE_RADIUS: f32 = 4000.0;

pub fn sdf_aur_spikes(p: Vec3, camera_offset: Vec2, textures: &Textures) -> f32 {
    let uv = (vec2(p.x, p.y) + camera_offset) * SPIKE_SCALE;

    let voronoi_dist = textures.extra(uv).x;

    let spike_top = SPIKE_BASE_ALT + BASE_UNDULATION + SPIKE_HEIGHT;
    let spike_bottom = SPIKE_BASE_ALT;

    let height_range = spike_top - spike_bottom;
    let relative_z = (p.z - spike_bottom) / height_range;
    let clamped_z = relative_z.saturate();
    let cone_radius = SPIKE_RADIUS * (1.0 - clamped_z);

    let current_r = voronoi_dist * SPIKE_RADIUS;

    let slope = SPIKE_RADIUS / height_range;
    let d_cone = (current_r - cone_radius) / (1.0 + slope * slope).sqrt();

    let d_vertical = p.z - spike_top;
    let d_bottom = spike_bottom - p.z;

    let d = d_cone.max(d_vertical).max(d_bottom);

    d * 0.5
}
