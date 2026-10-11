pub mod aur_spikes;

use spirv_std::glam::Vec3;

pub use crate::scene::surface::{Hit as ShadeResult, estimate_normal};

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
