use crate::utils::AtmosphereData;
use core::f32::consts::PI;
use spirv_std::{
    glam::{FloatExt, Vec3},
    num_traits::Float,
};

pub struct SurfaceLight {
    pub sun: Vec3,
    pub sky: Vec3,
    pub ambient: Vec3,
}

pub fn compute_surface_light(
    normal: Vec3,
    sun_dir: Vec3,
    sun_visibility: f32,
    atmosphere: &AtmosphereData,
) -> SurfaceLight {
    let n_dot_up = normal.z.max(0.0);
    let n_dot_sun = normal.dot(sun_dir).max(0.0);

    let sun = atmosphere.sun * sun_visibility * n_dot_sun;

    // Use ambient (hemispherical/zenith) rather than view-direction sky so
    // sunset horizon colour doesn't bleed into surface shading.
    let sky_factor = 0.4 + 0.6 * n_dot_up;
    let sky = atmosphere.ambient * sky_factor;

    let ambient = atmosphere.ambient * (0.5 + 0.5 * n_dot_up);

    SurfaceLight { sun, sky, ambient }
}

pub fn trace_sun_visibility<F: Fn(Vec3) -> f32>(
    pos: Vec3,
    sun_dir: Vec3,
    max_dist: f32,
    steps: u32,
    sdf: F,
) -> f32 {
    let step_size = max_dist / steps as f32;
    let mut visibility = 1.0;
    let mut t = step_size * 0.5;

    for _ in 0..steps {
        let p = pos + sun_dir * t;
        let d = sdf(p);
        if d < 0.01 {
            return 0.0;
        }
        let penumbra = (d / t).saturate();
        visibility = visibility.min(penumbra);
        t += step_size;
    }

    visibility
}

pub fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = (1.0 + g2 - 2.0 * g * cos_theta).powf(1.5);
    (1.0 - g2) / (4.0 * PI * denom.max(1e-4))
}
