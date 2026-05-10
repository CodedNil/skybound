mod aur_spikes;
mod ships;

use crate::lighting::{compute_surface_light, trace_sun_visibility};
use crate::solids::aur_spikes::{MAT_SPIKE, sdf_aur_spikes, spike_albedo};
use crate::solids::ships::{MAT_SPHERE, MAT_WING, mat_color, sdf_ships, shade_ship};
use crate::utils::{AtmosphereData, get_sun_position};
use skybound_shared::ViewUniform;
use spirv_std::glam::{FloatExt, Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

const MAX_STEPS: i32 = 512;
const EPSILON: f32 = 0.01;
const NORMAL_EPS: f32 = 0.02;
const MIN_STEP: f32 = 0.03;

const SUN_VISIBILITY_DIST: f32 = 50.0;
const SUN_VISIBILITY_STEPS: u32 = 12;
const REFLECTION_MAX_DIST: f32 = 6000.0;
const REFLECTION_STEPS: u32 = 256;

fn sdf_combined(p: Vec3, time: f32) -> (f32, u32) {
    let (d_creatures, mat_creatures) = sdf_ships(p, time);

    let d_spikes = sdf_aur_spikes(p);
    if d_spikes < d_creatures {
        (d_spikes, MAT_SPIKE)
    } else {
        (d_creatures, mat_creatures)
    }
}

fn sdf_dist(p: Vec3, time: f32) -> f32 {
    sdf_combined(p, time).0
}

fn estimate_normal(p: Vec3, time: f32) -> Vec3 {
    let dx = vec3(NORMAL_EPS, 0.0, 0.0);
    let dy = vec3(0.0, NORMAL_EPS, 0.0);
    let dz = vec3(0.0, 0.0, NORMAL_EPS);
    let nx = sdf_dist(p + dx, time) - sdf_dist(p - dx, time);
    let ny = sdf_dist(p + dy, time) - sdf_dist(p - dy, time);
    let nz = sdf_dist(p + dz, time) - sdf_dist(p - dz, time);
    vec3(nx, ny, nz).normalize()
}

fn trace_reflection(
    pos: Vec3,
    refl_dir: Vec3,
    time: f32,
    sun_dir: Vec3,
    atmosphere: &AtmosphereData,
) -> Vec3 {
    let step_size = REFLECTION_MAX_DIST / REFLECTION_STEPS as f32;
    let mut t = 0.1;

    for _ in 0..REFLECTION_STEPS {
        let p = pos + refl_dir * t;
        let (d, mat) = sdf_combined(p, time);
        if d < 0.05 {
            let n = estimate_normal(p, time);
            let n_dot_l = n.dot(sun_dir).max(0.0);
            let n_dot_up = n.z.max(0.0);
            let albedo = if mat == MAT_SPIKE {
                spike_albedo(p)
            } else {
                mat_color(mat)
            };
            let sun_vis =
                trace_sun_visibility(p, sun_dir, SUN_VISIBILITY_DIST, SUN_VISIBILITY_STEPS, |q| {
                    sdf_dist(q, time)
                });
            let diffuse = atmosphere.sun * n_dot_l * sun_vis * 0.7
                + atmosphere.ambient * (0.3 + 0.4 * n_dot_up);
            return albedo * diffuse;
        }
        t += d.max(step_size * 0.2);
        if t > REFLECTION_MAX_DIST {
            break;
        }
    }

    let sky_blend = (refl_dir.z * 0.5 + 0.5).saturate();
    atmosphere.sky * (0.3 + 0.7 * sky_blend) + atmosphere.ambient * 0.2
}

fn shade_basic(
    p: Vec3,
    n: Vec3,
    mat: u32,
    view: &ViewUniform,
    atmosphere: &AtmosphereData,
    time: f32,
) -> (Vec3, f32) {
    // Ships
    if mat == MAT_SPHERE || mat == MAT_WING {
        return shade_ship(
            p,
            n,
            mat,
            view,
            atmosphere,
            |q| sdf_dist(q, time),
            |pos, refl_dir| {
                let sun_pos = get_sun_position(view);
                let sun_dir = (sun_pos - pos).normalize();
                trace_reflection(pos, refl_dir, time, sun_dir, atmosphere)
            },
        );
    }

    // Spikes
    let sun_pos = get_sun_position(view);
    let sun_dir = (sun_pos - p).normalize();
    let sun_vis =
        trace_sun_visibility(p, sun_dir, SUN_VISIBILITY_DIST, SUN_VISIBILITY_STEPS, |q| {
            sdf_dist(q, time)
        });
    let light = compute_surface_light(n, sun_dir, sun_vis, atmosphere);
    let albedo = spike_albedo(p);
    let n_dot_up = n.z.max(0.0);
    let lit = light.sun + atmosphere.ambient * (0.2 + 0.4 * n_dot_up);
    let color = albedo * lit;

    (color, 0.0)
}

pub struct SolidsResult {
    pub color: Vec3,
    pub depth: f32,
    pub specular: f32,
    pub normal: Vec3,
    pub hit: f32,
}

pub fn raymarch_solids(
    ro: Vec3,
    rd: Vec3,
    view: &ViewUniform,
    atmosphere: &AtmosphereData,
    t_max: f32,
    dither: f32,
) -> SolidsResult {
    let time = view.time();
    let mut t = dither * 1.5;
    let mut out_color = Vec3::ZERO;
    let mut out_spec = 0.0;
    let mut out_normal = Vec3::ZERO;
    let mut out_depth = t_max;
    let mut hit = 0.0;

    for _ in 0..MAX_STEPS {
        if t >= t_max {
            break;
        }
        let p = ro + rd * t;
        let (dist, mat) = sdf_combined(p, time);
        if dist < EPSILON {
            let normal = estimate_normal(p, time);
            out_depth = t;
            hit = 1.0;
            if rd.dot(normal) <= 0.0 {
                let (color, specular) = shade_basic(p, normal, mat, view, atmosphere, time);
                out_color = color;
                out_spec = specular;
                out_normal = normal;
            }
            break;
        }
        t += dist.max(MIN_STEP);
    }

    SolidsResult {
        color: out_color.saturate(),
        depth: out_depth,
        specular: out_spec,
        normal: out_normal,
        hit,
    }
}
