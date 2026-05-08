use crate::utils::{AtmosphereData, View, get_sun_position};
use spirv_std::glam::Vec3;
use spirv_std::num_traits::Float;

const MAX_STEPS: i32 = 64;
const EPSILON: f32 = 0.01;
const MIN_STEP: f32 = 0.03;

const SPACING: f32 = 200.0;
const RADIUS: f32 = 2.0;

fn repeat_to_cell(p: Vec3, s: f32) -> Vec3 {
    let cell_center = (p / s + Vec3::splat(0.5)).floor() * s;
    p - cell_center
}

fn sdf_sphere(p: Vec3, r: f32) -> f32 {
    p.length() - r
}

fn estimate_normal(p: Vec3) -> Vec3 {
    let dx = Vec3::new(EPSILON, 0.0, 0.0);
    let dy = Vec3::new(0.0, EPSILON, 0.0);
    let dz = Vec3::new(0.0, 0.0, EPSILON);

    let local = repeat_to_cell(p, SPACING);
    let nx = sdf_sphere(local + dx, RADIUS) - sdf_sphere(local - dx, RADIUS);
    let ny = sdf_sphere(local + dy, RADIUS) - sdf_sphere(local - dy, RADIUS);
    let nz = sdf_sphere(local + dz, RADIUS) - sdf_sphere(local - dz, RADIUS);
    Vec3::new(nx, ny, nz).normalize()
}

pub struct ShadeResult {
    pub color: Vec3,
    pub spec: f32,
}

fn shade_basic(p: Vec3, n: Vec3, rd: Vec3, view: &View) -> ShadeResult {
    let base_color = Vec3::new(0.1, 0.8, 0.2);
    let sun_pos = get_sun_position(view);
    let sun_dir = (sun_pos - p).normalize();

    let lam = n.dot(sun_dir).max(0.0);
    let ambient = 0.12;
    let mut color = base_color * (ambient + lam * 0.9);

    let hemi_mask = if n.z > 0.0 { 1.0 } else { 0.0 };
    let spec = hemi_mask;

    color += Vec3::splat(spec);

    ShadeResult {
        color,
        spec: spec * hemi_mask,
    }
}

pub struct SolidsResult {
    pub color: Vec3,
    pub depth: f32,
    pub specular: f32,
    pub normal: Vec3,
}

pub fn raymarch_solids(ro: Vec3, rd: Vec3, view: &View, t_max: f32, _time: f32) -> SolidsResult {
    let mut t = 0.0;
    let mut out_color = Vec3::ZERO;
    let mut out_spec = 0.0;
    let mut out_normal = Vec3::ZERO;
    let mut out_depth = t_max;

    for _ in 0..MAX_STEPS {
        if t >= t_max {
            break;
        }

        let p = ro + rd * t;
        // let dist = sdf_repeating_spheres_world(p);
        let dist = 1.0;

        if dist < EPSILON {
            out_depth = t;
            out_normal = estimate_normal(p);
            let shade = shade_basic(p, out_normal, rd, view);
            out_color = shade.color;
            out_spec = shade.spec;
            break;
        }
        t += dist.max(MIN_STEP);
    }

    SolidsResult {
        color: out_color.clamp(Vec3::ZERO, Vec3::ONE),
        depth: out_depth,
        specular: out_spec,
        normal: out_normal,
    }
}
