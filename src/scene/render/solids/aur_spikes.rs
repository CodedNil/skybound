use crate::scene::{
    ViewUniform,
    render::solids::{ShadeResult, estimate_normal, trace_shadow, world_to_curved},
    terrain::{spike_distance, spike_interval},
};
use spirv_std::glam::{Vec3, Vec3Swizzles, Vec4Swizzles, vec3};

const MAX_STEPS: i32 = 256;
const EPSILON: f32 = 0.05;
const MIN_STEP: f32 = 0.05;

const SPIKE_COLOR: Vec3 = vec3(0.16, 0.09, 0.03);
pub fn raymarch_aur_spikes(ro: Vec3, rd: Vec3, view: &ViewUniform, t_max: f32) -> ShadeResult {
    let planet_center = view.planet_center();
    let camera_offset = view.camera_offset();
    let camera_offset_xy = camera_offset.xy();
    let interval = spike_interval(ro.z + camera_offset.z, rd, t_max);
    let end = interval.y;
    let mut t = interval.x;

    for _ in 0..MAX_STEPS {
        if t >= end {
            break;
        }
        let p_raw = ro + rd * t;
        let p = world_to_curved(p_raw, planet_center, camera_offset.z);
        let d = spike_distance(p, camera_offset_xy);
        if d < EPSILON {
            let normal = estimate_normal(p, |p| spike_distance(p, camera_offset_xy));
            if rd.dot(normal) <= 0.0 {
                let sun_pos = view.sun_position.xyz();
                let planet_center = view.planet_center();
                let light_dir = (sun_pos - (p + planet_center)).normalize();
                let dot_nl = normal.dot(light_dir).max(0.0);
                let shadow = if dot_nl > 0.0 {
                    trace_shadow(p, light_dir, 15000.0, |p| {
                        spike_distance(p, camera_offset_xy)
                    })
                } else {
                    0.0
                };
                return ShadeResult {
                    color_depth: (SPIKE_COLOR * (dot_nl * shadow + 0.05)).extend(t),
                    normal,
                    previous_position: p_raw,
                };
            }
            break;
        }
        t += d.max(MIN_STEP);
    }

    ShadeResult::MISS
}
