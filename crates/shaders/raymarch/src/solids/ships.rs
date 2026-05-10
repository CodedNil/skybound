use crate::lighting::{compute_surface_light, trace_sun_visibility};
use crate::utils::{AtmosphereData, get_sun_position};
use skybound_shared::ViewUniform;
use spirv_std::glam::{FloatExt, Vec3, Vec4Swizzles, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const SPACING: f32 = 250.0;
const RADIUS: f32 = 5.0;
const WING_HALF_SPAN: f32 = 4.0;
const WING_HALF_THICKNESS: f32 = 0.3;
const WING_HALF_CHORD: f32 = 2.5;

pub const MAT_SPHERE: u32 = 0;
pub const MAT_WING: u32 = 1;

const SUN_VISIBILITY_DIST: f32 = 50.0;
const SUN_VISIBILITY_STEPS: u32 = 12;

const WING_COLOR: Vec3 = vec3(0.4, 0.7, 1.0);
const SPHERE_COLOR: Vec3 = vec3(0.55, 0.52, 0.48);

pub const fn mat_color(mat: u32) -> Vec3 {
    if mat == MAT_WING {
        WING_COLOR
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

/// Returns (distance, material).
pub fn sdf_ships(p: Vec3, time: f32) -> (f32, u32) {
    let local = repeat_to_cell_xy(p, SPACING);
    let d_sphere = sdf_sphere(local, RADIUS);
    let d_wings = sdf_wings(local, time);
    let d = smin(d_sphere, d_wings, 1.0);
    let mat = if d_sphere < d_wings {
        MAT_SPHERE
    } else {
        MAT_WING
    };
    (d, mat)
}

pub fn shade_ship(
    p: Vec3,
    n: Vec3,
    mat: u32,
    view: &ViewUniform,
    atmosphere: &AtmosphereData,
    sdf_dist_fn: impl Fn(Vec3) -> f32,
    trace_refl_fn: impl Fn(Vec3, Vec3) -> Vec3,
) -> (Vec3, f32) {
    let sun_pos = get_sun_position(view);
    let sun_dir = (sun_pos - p).normalize();
    let sun_vis = trace_sun_visibility(
        p,
        sun_dir,
        SUN_VISIBILITY_DIST,
        SUN_VISIBILITY_STEPS,
        sdf_dist_fn,
    );
    let light = compute_surface_light(n, sun_dir, sun_vis, atmosphere);
    let view_dir = (view.world_position.xyz() - p).normalize();
    let refl_dir = 2.0 * n.dot(view_dir) * n - view_dir;
    let refl_color = trace_refl_fn(p, refl_dir);
    let n_dot_v = n.dot(view_dir).abs();

    let base = mat_color(mat) * (light.sun + light.sky * 0.5 + light.ambient * 0.8);

    if mat == MAT_SPHERE {
        // Sphere: strong mirror fresnel + sharp specular
        let fresnel = (1.0 - n_dot_v).powf(2.5);
        let color = base.lerp(refl_color, 0.75 * fresnel + 0.05);
        let half = (sun_dir + view_dir).normalize();
        let spec = n.dot(half).max(0.0).powf(256.0);
        (color + Vec3::splat(spec * sun_vis) * light.sun * 0.7, 1.0)
    } else {
        // Wing: diffuse rough reflection picks up environment (purple from below)
        let fresnel = 0.10 + 0.30 * (1.0 - n_dot_v).powf(3.0);
        let color = base.lerp(refl_color, fresnel);
        let n_dot_up = n.z.max(0.0);
        let gi = atmosphere.ambient * (0.3 + 0.3 * n_dot_up);
        (color + gi, 0.3)
    }
}
