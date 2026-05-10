use crate::utils::AtmosphereData;
use spirv_std::{
    glam::{FloatExt, Vec3},
    num_traits::Float,
};

/// Result of lighting a surface point.
pub struct SurfaceLight {
    /// Direct sunlight reaching this point.
    pub sun: Vec3,
    /// Skylight from the hemisphere above.
    pub sky: Vec3,
    /// Ambient / bounce light.
    pub ambient: Vec3,
}

/// Compute incident lighting at a surface point given a sun-visibility factor
/// (0 = fully occluded, 1 = fully lit), an aurora-visibility factor, and the
/// atmosphere data.
pub fn compute_surface_light(
    normal: Vec3,
    sun_dir: Vec3,
    sun_visibility: f32,
    atmosphere: &AtmosphereData,
) -> SurfaceLight {
    let n_dot_up = normal.z.max(0.0);
    let n_dot_sun = normal.dot(sun_dir).max(0.0);

    // Direct sun light attenuated by self-shadow visibility.
    let sun = atmosphere.sun * sun_visibility * n_dot_sun;

    // Sky light: stronger from above, weaker from below.
    let sky_factor = 0.4 + 0.6 * n_dot_up;
    let sky = atmosphere.sky * sky_factor;

    // Ambient: uniform base + bounce from sky.
    let ambient = atmosphere.ambient * (0.5 + 0.5 * n_dot_up);

    SurfaceLight { sun, sky, ambient }
}

/// March a short ray toward the sun through a signed distance field.
///
/// Returns visibility in [0, 1] — 1 = fully lit, 0 = fully occluded.
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
