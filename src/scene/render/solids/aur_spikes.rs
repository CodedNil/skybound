use spirv_std::glam::{Vec2, Vec3, Vec4Swizzles, vec3};

use crate::scene::ViewUniform;
use crate::scene::render::solids::{ShadeResult, estimate_normal, trace_shadow};
use crate::scene::terrain::{spike_distance, spike_interval};

const MAX_STEPS: i32 = 256;
const EPSILON: f32 = 0.05;
const MIN_STEP: f32 = 0.05;

const SPIKE_COLOR: Vec3 = vec3(0.16, 0.09, 0.03);
pub fn raymarch_aur_spikes(ro: Vec3, rd: Vec3, view: &ViewUniform, t_max: f32) -> ShadeResult {
    let planet_center = view.planet_center();
    let up = (ro - planet_center).normalize();
    let tangent = crate::scene::planet::surface_frame(up).transpose();
    let interval = spike_interval(view.surface_position(ro).z, tangent * rd, t_max);
    let end = interval.y;
    let mut t = interval.x;

    for _ in 0..MAX_STEPS {
        if t >= end {
            break;
        }
        let p_raw = ro + rd * t;
        let p = view.surface_position(p_raw);
        let d = spike_distance(p, Vec2::ZERO);
        if d < EPSILON {
            let normal = estimate_normal(p_raw, |p| {
                spike_distance(view.surface_position(p), Vec2::ZERO)
            });
            if rd.dot(normal) <= 0.0 {
                let pole_pos = view.pole_position.xyz();
                let light_dir = (pole_pos - p_raw).normalize();
                let dot_nl = normal.dot(light_dir).max(0.0);
                let shadow = if dot_nl > 0.0 {
                    trace_shadow(p_raw, light_dir, 15000.0, |p| {
                        spike_distance(view.surface_position(p), Vec2::ZERO)
                    })
                } else {
                    0.0
                };
                return ShadeResult {
                    color_depth: (SPIKE_COLOR * (dot_nl * shadow + 0.05)).extend(t),
                    normal,
                    previous_position: p_raw,
                    transmittance: Vec3::ZERO,
                    part: 0,
                };
            }
            break;
        }
        t += d.max(MIN_STEP);
    }

    ShadeResult::MISS
}
