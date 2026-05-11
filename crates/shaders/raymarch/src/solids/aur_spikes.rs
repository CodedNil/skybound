use crate::{
    solids::{ShadeResult, estimate_normal, trace_shadow, world_to_curved},
    utils::{Textures, get_sun_position},
};
use skybound_shared::ViewUniform;
use spirv_std::glam::{FloatExt, Vec2, Vec3, Vec3Swizzles, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

const MAX_STEPS: i32 = 256;
const EPSILON: f32 = 0.05;
const MIN_STEP: f32 = 0.05;

const SPIKE_COLOR: Vec3 = vec3(0.16, 0.09, 0.03);
const SPIKE_BASE_ALT: f32 = -6000.0;
const BASE_UNDULATION: f32 = 1500.0;
const SPIKE_HEIGHT: f32 = 4500.0;
const SPIKE_SCALE: f32 = 0.0001;

fn sdf_dist(p: Vec3, camera_offset: Vec2, textures: &Textures) -> f32 {
    let uv = (vec2(p.x, p.y) + camera_offset) * SPIKE_SCALE;
    let voronoi_val = textures.extra(uv).x;
    let spike_top = SPIKE_BASE_ALT + BASE_UNDULATION + SPIKE_HEIGHT * voronoi_val;
    let spike_bottom = SPIKE_BASE_ALT;
    let d_vertical = (p.z - spike_top).max(spike_bottom - p.z);
    let height_percent = ((p.z - spike_bottom) / (spike_top - spike_bottom + 0.1)).saturate();
    let radius = 150.0 * (1.0 - height_percent);
    let d_horizontal = (0.85 - voronoi_val) * (0.1 / SPIKE_SCALE) - radius;
    d_horizontal.max(d_vertical).max(0.0) * 0.5
}

pub fn raymarch_aur_spikes(
    ro: Vec3,
    rd: Vec3,
    view: &ViewUniform,
    t_max: f32,
    dither: f32,
    textures: &Textures,
) -> ShadeResult {
    let planet_center = view.planet_center();
    let camera_offset = view.camera_offset();
    let camera_offset_xy = camera_offset.xy();
    let mut t = dither * 1.5;

    for _ in 0..MAX_STEPS {
        if t >= t_max {
            break;
        }
        let p_raw = ro + rd * t;
        let p = world_to_curved(p_raw, planet_center, camera_offset.z);
        let d = sdf_dist(p, camera_offset_xy, textures);
        if d < EPSILON {
            let normal = estimate_normal(p, |p| sdf_dist(p, camera_offset_xy, textures));
            if rd.dot(normal) <= 0.0 {
                let sun_pos = get_sun_position(
                    view.planet_center(),
                    view.planet_rotation,
                    view.ro_relative(),
                    view.latitude(),
                );
                let planet_center = view.planet_center();
                let light_dir = (sun_pos - (p + planet_center)).normalize();
                let dot_nl = normal.dot(light_dir).max(0.0);
                let shadow = if dot_nl > 0.0 {
                    trace_shadow(p, light_dir, 15000.0, |p| {
                        sdf_dist(p, camera_offset_xy, textures)
                    })
                } else {
                    0.0
                };
                return ShadeResult {
                    color_depth: (SPIKE_COLOR * (dot_nl * shadow + 0.05)).extend(t),
                };
            }
            break;
        }
        t += d.max(MIN_STEP);
    }

    ShadeResult {
        color_depth: vec4(0.0, 0.0, 0.0, t_max),
    }
}
