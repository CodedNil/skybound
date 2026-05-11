pub mod aur_spikes;
pub mod ships;

use skybound_shared::PLANET_RADIUS;
use spirv_std::glam::{Vec3, Vec4, vec3};

pub const NORMAL_EPS: f32 = 0.1;

/// Surface shading result
pub struct ShadeResult {
    pub color_depth: Vec4,
}

pub fn world_to_curved(p_raw: Vec3, planet_center: Vec3, camera_offset_z: f32) -> Vec3 {
    let dx = p_raw.x - planet_center.x;
    let dy = p_raw.y - planet_center.y;
    let altitude = p_raw.z + camera_offset_z + (dx * dx + dy * dy) / (2.0 * PLANET_RADIUS);
    vec3(p_raw.x, p_raw.y, altitude)
}

pub fn trace_shadow<F>(pos: Vec3, light_dir: Vec3, max_dist: f32, mut sdf: F) -> f32
where
    F: FnMut(Vec3) -> f32,
{
    let mut t = 1.0;
    for _ in 0..32 {
        let p = pos + light_dir * t;
        let d = sdf(p);
        if d < 0.1 {
            return 0.0;
        }
        t += d.max(0.5);
        if t > max_dist {
            break;
        }
    }
    1.0
}

pub fn estimate_normal<F>(p: Vec3, mut sdf: F) -> Vec3
where
    F: FnMut(Vec3) -> f32,
{
    let dx = vec3(NORMAL_EPS, 0.0, 0.0);
    let dy = vec3(0.0, NORMAL_EPS, 0.0);
    let dz = vec3(0.0, 0.0, NORMAL_EPS);
    let nx = sdf(p + dx) - sdf(p - dx);
    let ny = sdf(p + dy) - sdf(p - dy);
    let nz = sdf(p + dz) - sdf(p - dz);
    let n = vec3(nx, ny, nz);
    if n.length_squared() > 1e-10 {
        n.normalize()
    } else {
        vec3(0.0, 0.0, 1.0)
    }
}
