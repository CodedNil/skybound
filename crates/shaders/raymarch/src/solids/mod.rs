mod aur_spikes;

use crate::lighting::{compute_surface_light, trace_sun_visibility};
use crate::solids::aur_spikes::{MAT_SPIKE, SPIKE_COLOR_MID, sdf_aur_spikes, spike_albedo};
use crate::utils::{AtmosphereData, get_sun_position};
use skybound_shared::ViewUniform;
use spirv_std::glam::{FloatExt, Vec3, Vec4Swizzles, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

const MAX_STEPS: i32 = 256;
const EPSILON: f32 = 0.01;
const NORMAL_EPS: f32 = 0.02;
const MIN_STEP: f32 = 0.03;

const SPACING: f32 = 250.0;
const RADIUS: f32 = 5.0;
const WING_HALF_SPAN: f32 = 4.0;
const WING_HALF_THICKNESS: f32 = 0.3;
const WING_HALF_CHORD: f32 = 2.5;

const MAT_SPHERE: u32 = 0;
const MAT_WING: u32 = 1;

const SUN_VISIBILITY_DIST: f32 = 50.0;
const SUN_VISIBILITY_STEPS: u32 = 12;
const REFLECTION_MAX_DIST: f32 = 40.0;
const REFLECTION_STEPS: u32 = 24;

const WING_COLOR: Vec3 = vec3(0.4, 0.7, 1.0);
const SPHERE_COLOR: Vec3 = vec3(0.55, 0.52, 0.48);

const fn mat_color(mat: u32) -> Vec3 {
    if mat == MAT_WING {
        WING_COLOR
    } else if mat == MAT_SPIKE {
        SPIKE_COLOR_MID
    } else {
        SPHERE_COLOR
    }
}

fn repeat_to_cell_xy(p: Vec3, s: f32) -> Vec3 {
    let cell_center = (p / s + Vec3::splat(0.5)).floor() * s;
    vec3(p.x - cell_center.x, p.y - cell_center.y, p.z)
}

fn sdf_sphere(p: Vec3, r: f32) -> f32 {
    p.length() - r
}

fn sdf_box(p: Vec3, half_extents: Vec3) -> f32 {
    let q = p.abs() - half_extents;
    q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
}

fn sdf_wing_single(p: Vec3, sign: f32, flap_angle: f32) -> f32 {
    let center = vec3(0.0, sign * (RADIUS + WING_HALF_SPAN), 0.0);
    let local = p - center;
    let span_t = ((local.y * sign + WING_HALF_SPAN) / (2.0 * WING_HALF_SPAN)).saturate();
    let bend = span_t * span_t * flap_angle * 0.8;
    let bent = vec3(local.x, local.y, local.z - bend);
    sdf_box(
        bent,
        vec3(WING_HALF_CHORD, WING_HALF_SPAN, WING_HALF_THICKNESS),
    )
}

fn sdf_wings(p: Vec3, time: f32) -> f32 {
    let flap_angle = (time * 3.0).sin() * 0.8;
    sdf_wing_single(p, 1.0, flap_angle).min(sdf_wing_single(p, -1.0, flap_angle))
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).saturate();
    b * (1.0 - h) + a * h - k * h * (1.0 - h)
}

fn sdf_combined(p: Vec3, time: f32) -> (f32, u32) {
    let local = repeat_to_cell_xy(p, SPACING);
    let d_sphere = sdf_sphere(local, RADIUS);
    let d_wings = sdf_wings(local, time);
    let d_creatures = smin(d_sphere, d_wings, 1.0);
    let mat_creatures = if d_sphere < d_wings {
        MAT_SPHERE
    } else {
        MAT_WING
    };

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
        if d < 0.02 {
            let n = estimate_normal(p, time);
            let n_dot_l = n.dot(sun_dir).max(0.0);
            let n_dot_up = n.z.max(0.0);
            let albedo = if mat == MAT_SPIKE {
                spike_albedo(p)
            } else {
                mat_color(mat)
            };
            let diffuse =
                atmosphere.sun * n_dot_l * 0.8 + atmosphere.ambient * (0.3 + 0.4 * n_dot_up);
            return albedo * diffuse;
        }
        t += d.max(step_size * 0.3);
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
    let sun_pos = get_sun_position(view);
    let sun_dir = (sun_pos - p).normalize();
    let sun_vis =
        trace_sun_visibility(p, sun_dir, SUN_VISIBILITY_DIST, SUN_VISIBILITY_STEPS, |q| {
            sdf_dist(q, time)
        });
    let light = compute_surface_light(n, sun_dir, sun_vis, atmosphere);
    let mut color = mat_color(mat) * (light.sun + light.sky * 0.5 + light.ambient * 0.8);

    if mat == MAT_SPHERE {
        let view_dir = (view.world_position.xyz() - p).normalize();
        let refl_dir = 2.0 * n.dot(view_dir) * n - view_dir;
        let refl_color = trace_reflection(p, refl_dir, time, sun_dir, atmosphere);

        let fresnel = (1.0 - n.dot(view_dir).abs()).powf(2.5);
        color = color.lerp(refl_color, 0.75 * fresnel + 0.05);

        let half = (sun_dir + view_dir).normalize();
        let spec = n.dot(half).max(0.0).powf(256.0);
        color += Vec3::splat(spec * sun_vis) * light.sun * 0.7;
    }

    if mat == MAT_WING {
        let view_dir = (view.world_position.xyz() - p).normalize();
        let n_dot_v = n.dot(view_dir).abs();
        let fresnel = 0.12 + 0.88 * (1.0 - n_dot_v).powf(5.0);
        let n_dot_up = n.z.max(0.0);
        let gi = atmosphere.ambient * (0.4 + 0.6 * n_dot_up);
        color += gi * fresnel;
    }

    if mat == MAT_SPIKE {
        let albedo = spike_albedo(p);
        let n_dot_up = n.z.max(0.0);
        let lit = light.sun + atmosphere.ambient * (0.2 + 0.4 * n_dot_up);
        color = albedo * lit;
    }

    let specular = if mat == MAT_SPHERE { 1.0 } else { 0.0 };
    (color, specular)
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
) -> SolidsResult {
    let time = view.time();
    let mut t = 0.0;
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
