use crate::utils::{MAGNETOSPHERE_HEIGHT, View, quat_rotate};
use spirv_std::glam::Vec3;
use spirv_std::num_traits::Float;

pub const POLE_WIDTH: f32 = 10000.0;

pub struct PolesSample {
    pub density: f32,
    pub color: Vec3,
    pub emission: Vec3,
}

pub fn sample_poles(pos: Vec3, _time: f32, _sampler: &spirv_std::Sampler) -> PolesSample {
    let mut color = Vec3::new(0.0, 0.5, 1.0);
    let atmosphere_dist =
        (5000.0 - (pos.z - MAGNETOSPHERE_HEIGHT).abs()).clamp(0.0, 4900.0) / 4900.0;
    if atmosphere_dist > 0.0 {
        color += Vec3::splat(atmosphere_dist.smoothstep(0.0, 1.0));
    }

    PolesSample {
        density: 1.0,
        color,
        emission: color,
    }
}

pub fn poles_raymarch_entry(ro: Vec3, rd: Vec3, view: &View, t_max: f32) -> (f32, f32) {
    let axis = quat_rotate(view.planet_rotation, Vec3::new(0.0, 0.0, 1.0)).normalize();
    let oc = ro - view.planet_center;
    let ad = axis.dot(rd);
    let ao = axis.dot(oc);
    let a = 1.0 - ad * ad;
    let b = 2.0 * (oc.dot(rd) - ao * ad);
    let c = oc.dot(oc) - ao * ao - POLE_WIDTH * POLE_WIDTH;

    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 || a.abs() < 1e-6 {
        return (t_max, 0.0);
    }

    let s = disc.sqrt();
    let t0 = (-b - s) / (2.0 * a);
    let t1 = (-b + s) / (2.0 * a);

    let entry = t0.min(t1);
    let exit = t0.max(t1);

    if exit <= 0.0 {
        return (t_max, 0.0);
    }
    (entry, exit)
}

trait Smoothstep {
    fn smoothstep(self, edge0: Self, edge1: Self) -> Self;
}

impl Smoothstep for f32 {
    fn smoothstep(self, edge0: f32, edge1: f32) -> f32 {
        let t = ((self - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }
}
