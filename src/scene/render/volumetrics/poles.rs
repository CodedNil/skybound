use super::VolumeSample;
use crate::scene::{ViewUniform, render::utils::MAGNETOSPHERE_HEIGHT};
use spirv_std::glam::{FloatExt, Vec2, Vec3, vec2, vec3};
use spirv_std::num_traits::Float;

pub const POLE_WIDTH: f32 = 10000.0;

pub fn sample_poles(pos: Vec3) -> VolumeSample {
    let mut color = vec3(0.0, 0.5, 1.0);

    let atmosphere_dist = (pos.z - MAGNETOSPHERE_HEIGHT)
        .abs()
        .smoothstep(5000.0, 100.0);
    if atmosphere_dist > 0.0 {
        color += Vec3::splat(atmosphere_dist);
    }

    VolumeSample {
        density: 1.0,
        color,
        emission: color,
    }
}

pub fn poles_raymarch_entry(ro: Vec3, rd: Vec3, view: &ViewUniform, t_max: f32) -> Vec2 {
    let axis = view.planet_rotation.mul_vec3(Vec3::Z).normalize();
    let oc = ro - view.planet_center();
    let ad = axis.dot(rd);
    let ao = axis.dot(oc);
    let a = 1.0 - ad * ad;
    let b = 2.0 * (oc.dot(rd) - ao * ad);
    let c = oc.dot(oc) - ao * ao - POLE_WIDTH * POLE_WIDTH;

    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 || a == 0.0 {
        return vec2(t_max, 0.0);
    }

    let s = disc.sqrt();
    let t0 = (-b - s) / (2.0 * a);
    let t1 = (-b + s) / (2.0 * a);

    let entry = t0.min(t1);
    let exit = t0.max(t1);

    if exit <= 0.0 {
        return vec2(t_max, 0.0);
    }
    vec2(entry, exit)
}
